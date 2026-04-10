import os
import gc
import sys
import json
import torch
import warnings
import argparse
import numpy as np
import torch.multiprocessing as mp
from copy import deepcopy

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="WSI-VQA Multi-GPU Inference Pipeline")

    # --- Library Paths ---
    parser.add_argument("--plip_lib_path", type=str, required=True,
                        help="Path to the PLIP library directory")

    # --- Model Checkpoints ---
    parser.add_argument("--qwen_ckpt", type=str, required=True,
                        help="Path to Qwen checkpoint")
    parser.add_argument("--plip_ckpt", type=str, required=True,
                        help="Path to PLIP checkpoint")
    parser.add_argument("--patho_r1_ckpt", type=str, required=True,
                        help="Path to Patho-R1 checkpoint")

    # --- Data Files ---
    parser.add_argument("--descriptions_file", type=str, required=True,
                        help="Path to patch descriptions JSON file")

    # 同时兼容 questions_file / test_questions_file
    parser.add_argument("--questions_file", "--test_questions_file",
                        dest="questions_file", type=str, required=True,
                        help="Path to questions/VQA dataset JSON file")

    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory containing image features")
    parser.add_argument("--patch_root", type=str, required=True,
                        help="Root directory for image patches")

    # 同时兼容 save_dir / test_save_dir
    parser.add_argument("--save_dir", "--test_save_dir",
                        dest="save_dir", type=str, required=True,
                        help="Directory to save results")

    # --- Settings ---
    parser.add_argument("--dataset_name", type=str, default="wsi_vqa",
                        help="Name of the dataset (e.g., wsi_vqa, slidechat)")

    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated GPU ids, e.g. '0,1,2,3'")
    parser.add_argument("--procs_per_gpu", type=int, default=1,
                        help="Number of worker processes to launch per GPU")
    parser.add_argument("--initial_sample_ratio", type=float, default=0.10)
    parser.add_argument("--replenish_ratio", type=float, default=0.05)
    parser.add_argument("--random_seed", type=int, default=128)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--zoom_level_val", type=int, default=5)

    return parser.parse_args()


args = parse_args()

if not os.path.exists(args.plip_lib_path):
    raise FileNotFoundError(f"PLIP library path not found: {args.plip_lib_path}")

sys.path.insert(0, args.plip_lib_path)

from plip import PLIP
from data_processing.utils import *
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from models.inference import (
    evaluate_with_llm_chain,
    slide_llm_answer,
    patho_r1_describe,
    summarize_patches_in_chunks,
)


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_torch_dtype():
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def move_plip_to_device(plip, device):
    """
    尽量兼容不同 PLIP 封装方式。
    """
    if hasattr(plip, "model"):
        plip.model = plip.model.to(device)
        plip.model.eval()
    elif hasattr(plip, "to"):
        plip = plip.to(device)
        if hasattr(plip, "eval"):
            plip.eval()
    return plip


def load_models_for_device(args, device):
    dtype = get_torch_dtype()

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_ckpt,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()

    plip = PLIP(args.plip_ckpt)
    plip = move_plip_to_device(plip, device)

    patho_r1_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.patho_r1_ckpt,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    patho_r1_model.eval()

    patho_r1_processor = AutoProcessor.from_pretrained(args.patho_r1_ckpt)

    return tokenizer, model, plip, patho_r1_model, patho_r1_processor


def process_single_case(
    pair,
    args,
    tokenizer,
    model,
    plip,
    patho_r1_model,
    patho_r1_processor,
    device,
    all_descriptions,
    rank,
):
    TARGET_LONG_ID = pair["long_id"]
    all_keys = list(all_descriptions.keys())

    def resolve_long_id(target_long_id, keys):
        # 1) exact
        if target_long_id in all_descriptions:
            return target_long_id, "exact"
        target_long_id = str(target_long_id)
        # 2) key starts with target (existing behavior)
        start_matches = [k for k in keys if k.startswith(target_long_id)]
        if start_matches:
            return start_matches[0], "key_startswith_target"

        # 3) target starts with key (handle shorter key in description file)
        reverse_start_matches = [k for k in keys if target_long_id.startswith(k)]
        if reverse_start_matches:
            return reverse_start_matches[0], "target_startswith_key"

        # 4) core-id match before '.' as a fallback
        target_core = target_long_id.split(".")[0]
        core_matches = [k for k in keys if k.split(".")[0] == target_core]
        if core_matches:
            return core_matches[0], "core_id_match"

        return target_long_id, None

    resolved_long_id, match_mode = resolve_long_id(TARGET_LONG_ID, all_keys)
    if match_mode is not None and resolved_long_id != TARGET_LONG_ID:
        print(f"[GPU {rank}] Resolved long_id by {match_mode}: '{TARGET_LONG_ID}' -> '{resolved_long_id}'")
    elif match_mode is None:
        print(f"[GPU {rank}] Target ID '{TARGET_LONG_ID}' not found in description index.")

    TARGET_LONG_ID = resolved_long_id

    question_text = pair["question"]
    question_choices = pair["choices"]
    correct_answer = pair["answer"]

    unique_id = make_unique_id(TARGET_LONG_ID, question_text)
    save_path = os.path.join(args.save_dir, f"{unique_id}.json")

    if os.path.exists(save_path):
        print(f"[GPU {rank}] Skipping: {unique_id} (Already completed)")
        return None

    print(f"\n[GPU {rank}] Starting Case: {unique_id}")
    print("=" * 80)
    print(f"[GPU {rank}] Question: {question_text}")
    print(f"[GPU {rank}] Choices: {question_choices}")
    print(f"[GPU {rank}] Ground Truth: {correct_answer}")

    descriptions_dict = all_descriptions.get(TARGET_LONG_ID)
    if not isinstance(descriptions_dict, dict) or len(descriptions_dict) == 0:
        print(f"[GPU {rank}] Missing or invalid descriptions for '{TARGET_LONG_ID}', skipping case.")
        return None

    descriptions_dict = deepcopy(descriptions_dict)
    print(f"[GPU {rank}] Extracted {len(descriptions_dict)} patch descriptions.")

    feature_cache = {}
    all_patch_names = list(descriptions_dict.keys())
    for patch_name in all_patch_names:
        npy_path = os.path.join(args.feature_dir, TARGET_LONG_ID, f"{patch_name.split('.')[0]}.npy")
        if os.path.exists(npy_path):
            feature_cache[patch_name] = np.load(npy_path)

    print(f"[GPU {rank}] Successfully cached {len(feature_cache)} / {len(all_patch_names)} patch features")

    q_emb = plip.encode_text([question_text], batch_size=1)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=-1, keepdims=True)

    available_patches = [p for p in all_patch_names if p in feature_cache]
    if not available_patches:
        print(f"[GPU {rank}] No available patches with features found. Skipping case.")
        return None

    feats = np.stack([feature_cache[p] for p in available_patches], axis=0)
    sims_all = (feats @ q_emb.T).squeeze()

    K = max(1, int(len(all_patch_names) * args.initial_sample_ratio))
    topk_idx = np.argsort(sims_all)[-K:][::-1]
    initial_sample_names = [available_patches[i] for i in topk_idx]

    accumulated_patch_names = list(initial_sample_names)
    patches_to_evaluate_names = list(initial_sample_names)
    remaining_patch_names = [p for p in all_patch_names if p not in accumulated_patch_names]

    print(f"[GPU {rank}] Initial sample size: {len(initial_sample_names)}; Remaining: {len(remaining_patch_names)}")

    attempt = 0
    final_answer = None
    all_results = []
    zoom_level_val = args.zoom_level_val

    while attempt < args.max_attempts:
        attempt += 1
        print(f"\n[GPU {rank}] Loop attempt {attempt}... Evaluating {len(patches_to_evaluate_names)} patches this round.")

        if not patches_to_evaluate_names:
            print(f"[GPU {rank}] No new patches available for evaluation this round, stopping.")
            break

        num_top_patches_to_describe = min(5, len(patches_to_evaluate_names))
        for i in range(num_top_patches_to_describe):
            patch_name = patches_to_evaluate_names[i]
            patch_path = get_patch_fullpath(args.patch_root, TARGET_LONG_ID, patch_name)
            patch_img = load_image(patch_path)

            x, y = extract_coords_from_name(patch_name)
            question_specific_desc = patho_r1_describe(
                patch_img,
                question=question_text,
                patho_r1_processor=patho_r1_processor,
                patho_r1_model=patho_r1_model,
                coords=(x, y),
                magnification=zoom_level_val,
                choices=question_choices
            )

            orig = descriptions_dict.get(patch_name, "")
            descriptions_dict[patch_name] = (orig + " " + question_specific_desc).strip()
            print(f"[GPU {rank}] Appended question-specific description for {patch_name} "
                  f"(Total length {len(descriptions_dict[patch_name])})")

        current_items_to_evaluate = [(name, descriptions_dict[name]) for name in patches_to_evaluate_names]
        desc_for_evaluation = build_descriptions_with_meta(
            current_items_to_evaluate,
            mag_level=zoom_level_val,
            include_header=True,
            include_coords=True
        )

        evaluation_result = evaluate_with_llm_chain(
            model, tokenizer, desc_for_evaluation, question_text, question_choices
        )
        can_now = str(evaluation_result.get("sufficient", "")).strip().lower() == "yes"
        can_zoom = str(evaluation_result.get("zoom_recommendation", "")).strip().lower() == "yes"

        if can_now:
            print(f"\n[GPU {rank}] patch-llm judgment: Can answer now.")
            final_desc_text = summarize_patches_in_chunks(
                model, tokenizer,
                descriptions_dict, accumulated_patch_names,
                question_text=question_text,
                chunk_size=10, threshold=50,
                magnification=zoom_level_val
            )
            final_answer = slide_llm_answer(
                model, tokenizer,
                final_desc_text,
                question_text,
                question_choices,
                magnification=zoom_level_val,
                case_name=unique_id,
            )

            all_results = [{
                "attempt": attempt,
                "mode": "can_answer_now",
                "evaluated_patches_this_round": patches_to_evaluate_names,
                "total_accumulated_patches": len(accumulated_patch_names),
                "evaluation_result": evaluation_result,
                "answer": final_answer["answer"],
                "explanation": final_answer["explanation"]
            }]
            break

        elif can_zoom:
            final_desc_text = summarize_patches_in_chunks(
                model, tokenizer,
                descriptions_dict, accumulated_patch_names,
                question_text=question_text,
                chunk_size=10, threshold=50,
                magnification=zoom_level_val
            )
            zoom_reason = evaluation_result.get("zoom_reason", "").strip()
            print(f"\n[GPU {rank}] patch-evaluator judgment: Need zoom. Reason: {zoom_reason}")

            zoom_level_raw = evaluation_result.get("zoom_level", zoom_level_val)
            try:
                zoom_level_val = int(zoom_level_raw)
            except (ValueError, TypeError):
                pass

            print(f"[GPU {rank}] -> Zoom level: {zoom_level_val}x")

            patch_feats = []
            patch_names_valid = []
            for p in patches_to_evaluate_names:
                if p in feature_cache:
                    patch_feats.append(feature_cache[p])
                    patch_names_valid.append(p)

            if not patch_feats:
                print(f"[GPU {rank}] No valid patch features, skipping zoom process.")
            else:
                patch_feats = np.stack(patch_feats, axis=0)
                text_emb = plip.encode_text([question_text], batch_size=1)
                text_emb = text_emb / np.linalg.norm(text_emb, axis=-1, keepdims=True)
                sims = np.dot(patch_feats, text_emb.T).squeeze()

                top2_idx = np.argsort(sims)[-2:][::-1]
                top2_names = [patch_names_valid[i] for i in top2_idx]
                print(f"[GPU {rank}] -> Selected Top2 patches: {top2_names}")

                candidate_images = []
                candidate_names = []
                for patch_name in top2_names:
                    patch_path = get_patch_fullpath(args.patch_root, TARGET_LONG_ID, patch_name)
                    sub_patches = split_patch_for_zoom(patch_path, zoom_level_val)
                    for sub_img, (x, y) in sub_patches:
                        candidate_images.append(sub_img)
                        candidate_names.append(f"{x}_{y}")

                if not candidate_images:
                    print(f"[GPU {rank}] No sub-patches generated, skipping zoom process.")
                else:
                    image_embs = plip.encode_images(candidate_images, batch_size=4)
                    image_embs = image_embs / np.linalg.norm(image_embs, axis=-1, keepdims=True)
                    sims_sub = np.dot(image_embs, text_emb.T).squeeze()
                    best_idx = int(np.argmax(sims_sub))
                    best_img = candidate_images[best_idx]
                    best_name = candidate_names[best_idx]
                    print(f"[GPU {rank}] Selected zoomed sub-patch: {best_name}")

                    x, y = best_name.split("_")

                    zoomed_patch_desc = patho_r1_describe(
                        best_img,
                        question=question_text,
                        patho_r1_processor=patho_r1_processor,
                        patho_r1_model=patho_r1_model,
                        coords=(x, y),
                        magnification=zoom_level_val,
                        choices=question_choices
                    )

                    extra_note = (
                        f"\n\n[NOTE] A key sub-patch was identified after zooming.\n"
                        f"Patch=({x},{y}), Magnification={zoom_level_val}x\n"
                        f"Detail: {zoomed_patch_desc}"
                    )
                    final_desc_text = final_desc_text + extra_note

                    final_answer = slide_llm_answer(
                        model, tokenizer,
                        final_desc_text,
                        question_text, question_choices,
                        magnification=zoom_level_val,
                        case_name=unique_id,
                    )

                    all_results = [{
                        "attempt": attempt,
                        "mode": "zoom_then_select",
                        "evaluated_patches_this_round": patches_to_evaluate_names,
                        "total_accumulated_patches": len(accumulated_patch_names),
                        "evaluation_result": evaluation_result,
                        "selected_zoom_patch": best_name,
                        "zoom_patch_desc": zoomed_patch_desc,
                        "answer": final_answer["answer"],
                        "explanation": final_answer["explanation"]
                    }]
                    break

        else:
            missing_info = evaluation_result.get("missing_info", "pathology details").strip()
            print(f"[GPU {rank}] patch-evaluator identifies missing info: '{missing_info}' -> Searching for new patches.")

            if not remaining_patch_names:
                print(f"[GPU {rank}] No more remaining patches available, loop terminated.")
                break

            valid_remaining_names = [name for name in remaining_patch_names if name in feature_cache]
            if not valid_remaining_names:
                print(f"[GPU {rank}] No valid feature cache for remaining patches, loop terminated.")
                break

            remaining_embs = np.stack([feature_cache[name] for name in valid_remaining_names], axis=0)
            text_emb = plip.encode_text([missing_info], batch_size=1)
            text_emb = text_emb / np.linalg.norm(text_emb, axis=-1, keepdims=True)
            sims = np.dot(remaining_embs, text_emb.T).squeeze()

            num_to_add = max(1, int(len(all_patch_names) * args.replenish_ratio))
            best_indices = np.argsort(sims)[::-1][:num_to_add]
            newly_selected_patches = [valid_remaining_names[i] for i in best_indices]
            print(f"[GPU {rank}] Selected {len(newly_selected_patches)} new relevant patches.")

            patches_to_evaluate_names = newly_selected_patches
            accumulated_patch_names.extend(newly_selected_patches)
            remaining_patch_names = [p for p in remaining_patch_names if p not in newly_selected_patches]

            print(f"[GPU {rank}] Total accumulated patches: {len(accumulated_patch_names)}, "
                  f"Remaining available: {len(remaining_patch_names)}")

    if final_answer is None:
        print(f"[GPU {rank}] Answer not found, entering fallback...")
        final_desc_text = summarize_patches_in_chunks(
            model, tokenizer,
            descriptions_dict, accumulated_patch_names,
            question_text=question_text,
            chunk_size=10, threshold=50,
            magnification=zoom_level_val
        )
        final_answer = slide_llm_answer(
            model, tokenizer,
            final_desc_text,
            question_text,
            question_choices,
            magnification=zoom_level_val,
            case_name=unique_id,
        )

        all_results.append({
            "attempt": "final_fallback",
            "mode": "fallback_final_attempt",
            "total_accumulated_patches": len(accumulated_patch_names),
            "answer": final_answer["answer"],
            "explanation": final_answer["explanation"]
        })

    print(f"[GPU {rank}] single_case_result: {final_answer['answer']}")
    print(f"[GPU {rank}] ground_truth: {correct_answer}")

    case_result = {
        "long_id": TARGET_LONG_ID,
        "question": question_text,
        "choices": question_choices,
        "ground_truth": correct_answer,
        "pred_answer": final_answer["answer"],
        "explanation": final_answer["explanation"],
        "process": all_results
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(case_result, f, indent=2, ensure_ascii=False)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return case_result


def worker_main(rank, worker_gpu_ids, args):
    gpu_id = worker_gpu_ids[rank]

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    seed_everything(args.random_seed + rank)
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 100)
    print(f"[Worker {rank}] Bound to GPU {gpu_id}, device = {device}")
    print(f"[Worker {rank}] Loading models...")
    print("=" * 100)

    tokenizer, model, plip, patho_r1_model, patho_r1_processor = load_models_for_device(args, device)

    print(f"[Worker {rank}] Models loaded successfully.")

    pairs = load_all_vqa_pairs(
        args.questions_file,
        dataset_name=args.dataset_name,
        image_dir=args.patch_root
    )
    print(f"[Worker {rank}] Successfully loaded {len(pairs)} VQA pairs.")

    with open(args.descriptions_file, "r", encoding="utf-8") as f:
        all_descriptions = json.load(f)
    print(f"[Worker {rank}] Pre-loaded descriptions file, containing {len(all_descriptions)} keys.")

    total_workers = len(worker_gpu_ids)
    local_pairs = [pair for idx, pair in enumerate(pairs) if idx % total_workers == rank]
    print(f"[Worker {rank}] Assigned {len(local_pairs)} cases.")

    local_results = []
    for pair in local_pairs:
        try:
            result = process_single_case(
                pair=pair,
                args=args,
                tokenizer=tokenizer,
                model=model,
                plip=plip,
                patho_r1_model=patho_r1_model,
                patho_r1_processor=patho_r1_processor,
                device=device,
                all_descriptions=all_descriptions,
                rank=rank,
            )
            if result is not None:
                local_results.append(result)
        except Exception as e:
            print(f"[Worker {rank}] Error while processing case: {e}")
            import traceback
            traceback.print_exc()

    summary_path = os.path.join(args.save_dir, f"_worker_{rank}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(local_results, f, indent=2, ensure_ascii=False)

    print(f"[Worker {rank}] Finished. Saved worker summary to {summary_path}")



def merge_results(save_dir):
    all_case_results = []
    for f in os.listdir(save_dir):
        if f.endswith(".json") and not f.startswith("_worker_"):
            fp = os.path.join(save_dir, f)
            try:
                with open(fp, "r", encoding="utf-8") as rf:
                    all_case_results.append(json.load(rf))
            except Exception as e:
                print(f"[Merge] Failed to read {fp}: {e}")

    final_path = os.path.join(save_dir, "all_results.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(all_case_results, f, indent=2, ensure_ascii=False)

    print("\n\n========== All Results ==========")
    print(json.dumps(all_case_results[:5], indent=2, ensure_ascii=False))
    print(f"\n[Merge] Total cases merged: {len(all_case_results)}")
    print(f"[Merge] Final merged file saved to: {final_path}")


def main():
    os.makedirs(args.save_dir, exist_ok=True)

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip() != ""]
    if len(gpu_ids) == 0:
        raise ValueError("No valid gpu_ids provided.")
    if args.procs_per_gpu < 1:
        raise ValueError("procs_per_gpu must be >= 1.")

    # Expand to one entry per worker; repeated gpu_id means multiple workers share one GPU.
    worker_gpu_ids = []
    for gid in gpu_ids:
        worker_gpu_ids.extend([gid] * args.procs_per_gpu)
    total_workers = len(worker_gpu_ids)

    print("=" * 60)
    print(f"PLIP Lib:   {args.plip_lib_path}")
    print(f"Model Qwen: {args.qwen_ckpt}")
    print(f"Model PLIP: {args.plip_ckpt}")
    print(f"Model PR1:  {args.patho_r1_ckpt}")
    print(f"Dataset:    {args.dataset_name}")
    print(f"Questions:  {args.questions_file}")
    print(f"Results to: {args.save_dir}")
    print(f"GPU IDs:    {gpu_ids}")
    print(f"Proc/GPU:   {args.procs_per_gpu}")
    print(f"Workers:    {total_workers}")
    print("=" * 60)

    mp.set_start_method("spawn", force=True)
    mp.spawn(worker_main, args=(worker_gpu_ids, args), nprocs=total_workers, join=True)

    merge_results(args.save_dir)


if __name__ == "__main__":
    main()
