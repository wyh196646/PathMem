import os
import gc
import re
import sys
import json
import time
import math
import uuid
import fcntl
import torch
import warnings
import argparse
import numpy as np
import torch.multiprocessing as mp
from copy import deepcopy

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="WSI-VQA Multi-GPU Inference Pipeline with Memory Bank (Training-Free Self-Evolving Agent)"
    )

    # --------------------------
    # Paths / checkpoints
    # --------------------------
    parser.add_argument("--plip_lib_path", type=str, required=True,
                        help="Path to the PLIP library directory")
    parser.add_argument("--qwen_ckpt", type=str, required=True,
                        help="Path to Qwen checkpoint")
    parser.add_argument("--plip_ckpt", type=str, required=True,
                        help="Path to PLIP checkpoint")
    parser.add_argument("--patho_r1_ckpt", type=str, required=True,
                        help="Path to Patho-R1 checkpoint")

    # --------------------------
    # Data
    # --------------------------
    parser.add_argument("--descriptions_file", type=str, required=True,
                        help="Path to patch descriptions JSON file")

    parser.add_argument("--questions_file", "--test_questions_file",
                        dest="questions_file", type=str, required=True,
                        help="Path to questions/VQA dataset JSON file")

    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory containing image features")
    parser.add_argument("--patch_root", type=str, required=True,
                        help="Root directory for image patches")
    parser.add_argument("--save_dir", "--test_save_dir",
                        dest="save_dir", type=str, required=True,
                        help="Directory to save results")

    # --------------------------
    # Runtime mode
    # --------------------------
    parser.add_argument("--mode", type=str, default="eval",
                        choices=["build_memory", "eval"],
                        help="build_memory: use train split to build/update memory; eval: read-only test inference")

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

    # --------------------------
    # Global summary / retrieval
    # --------------------------
    parser.add_argument("--global_summary_chunk_size", type=int, default=24)
    parser.add_argument("--global_summary_threshold", type=int, default=120)
    parser.add_argument("--global_summary_cache_dir", type=str, default=None)

    parser.add_argument("--visual_weight", type=float, default=0.75)
    parser.add_argument("--desc_weight", type=float, default=0.25)
    parser.add_argument("--num_question_specific_desc", type=int, default=5)
    parser.add_argument("--max_keywords", type=int, default=64)

    # --------------------------
    # Memory bank
    # --------------------------
    parser.add_argument("--memory_bank_path", type=str, default=None,
                        help="Path to memory bank JSON file; default: save_dir/memory_bank.json")
    parser.add_argument("--memory_enabled", action="store_true",
                        help="Enable memory retrieval/injection")
    parser.add_argument("--memory_lambda", type=float, default=0.5,
                        help="Blend weight between intent similarity and utility")
    parser.add_argument("--memory_topk_phase1", type=int, default=12,
                        help="Phase-1 candidate recall")
    parser.add_argument("--memory_topk_phase2", type=int, default=3,
                        help="Phase-2 final selected memories")
    parser.add_argument("--memory_sim_threshold", type=float, default=0.15,
                        help="Intent similarity threshold for memory recall")
    parser.add_argument("--memory_update_alpha", type=float, default=0.3,
                        help="EMA update rate for memory utility")
    parser.add_argument("--memory_max_context_chars", type=int, default=3000,
                        help="Max memory context chars injected into prompts")
    parser.add_argument("--memory_max_items", type=int, default=50000,
                        help="Soft cap for memory items")
    parser.add_argument("--memory_use_global_summary_in_intent", action="store_true",
                        help="Whether to include slide global summary in memory intent text")

    # --------------------------
    # Reward
    # --------------------------
    parser.add_argument("--reward_correct", type=float, default=1.0)
    parser.add_argument("--reward_wrong", type=float, default=0.0)

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


# =========================================================
# Utility
# =========================================================
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
    if hasattr(plip, "model"):
        plip.model = plip.model.to(device)
        plip.model.eval()
    elif hasattr(plip, "to"):
        plip = plip.to(device)
        if hasattr(plip, "eval"):
            plip.eval()
    return plip


def safe_l2_normalize(x, axis=-1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def zscore(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    mu = arr.mean()
    sigma = arr.std()
    if sigma < 1e-8:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def truncate_text(text, max_chars):
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "from",
    "what", "which", "is", "are", "was", "were", "this", "that", "these", "those",
    "most", "likely", "best", "least", "does", "do", "did", "by", "at", "as",
    "be", "been", "being", "into", "about", "than", "it", "its", "their", "his",
    "her", "if", "then", "than", "based", "find", "identify", "diagnosis", "diagnostic",
    "option", "options", "choice", "choices", "answer", "question", "slide", "wsi",
    "patch", "region", "tissue"
}


def simple_tokenize(text):
    if not text:
        return []
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    return tokens


def extract_keywords(text, max_keywords=64):
    tokens = simple_tokenize(text)
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in items[:max_keywords]]


def infer_task_type(question_text, choices):
    q = (question_text + " " + " ".join(str(choice) for choice in choices)).lower()
    if any(k in q for k in ["grade", "grading", "degree", "score"]):
        return "grading"
    if any(k in q for k in ["subtype", "type", "variant", "classify", "classification"]):
        return "subtyping"
    if any(k in q for k in ["stage", "staging", "metastasis", "invasion"]):
        return "staging_or_invasion"
    if any(k in q for k in ["receptor", "her2", "er", "pr", "biomarker"]):
        return "biomarker_related"
    if any(k in q for k in ["necrosis", "mitosis", "pleomorphism", "nuclear", "gland", "stroma"]):
        return "morphology_evidence"
    return "diagnosis"


# =========================================================
# Global summary
# =========================================================
def build_slide_level_global_summary(
    model,
    tokenizer,
    descriptions_dict,
    target_long_id,
    cache_dir,
    chunk_size=24,
    threshold=120,
):
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, f"{target_long_id}.json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            summary = obj.get("global_summary", "").strip()
            if summary:
                return summary
        except Exception:
            pass

    all_patch_names = list(descriptions_dict.keys())
    if len(all_patch_names) == 0:
        return ""

    generic_question = (
        "Please summarize the whole-slide pathology morphology using all available patch descriptions. "
        "Focus on dominant tissue patterns, lesion-rich regions, tumor/stroma distribution, gland formation, "
        "nuclear atypia, mitotic activity, necrosis, inflammation, fibrosis, and other discriminative findings "
        "that would help downstream patch retrieval and VQA reasoning."
    )

    try:
        summary = summarize_patches_in_chunks(
            model, tokenizer,
            descriptions_dict,
            all_patch_names,
            question_text=generic_question,
            chunk_size=chunk_size,
            threshold=threshold,
            magnification=5
        )
        summary = summary.strip()
    except Exception as e:
        print(f"[GlobalSummary] summarize_patches_in_chunks failed for {target_long_id}: {e}")
        sample_items = list(descriptions_dict.items())[: min(50, len(descriptions_dict))]
        merged = []
        for patch_name, desc in sample_items:
            if isinstance(desc, str) and desc.strip():
                merged.append(f"{patch_name}: {desc.strip()}")
        summary = "\n".join(merged)[:6000]

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {"long_id": target_long_id, "global_summary": summary},
                f,
                indent=2,
                ensure_ascii=False
            )
    except Exception as e:
        print(f"[GlobalSummary] Failed to save cache for {target_long_id}: {e}")

    return summary


def prepend_global_summary_to_text(slide_summary, raw_text):
    if not slide_summary or not slide_summary.strip():
        return raw_text
    prefix = (
        "[WHOLE-SLIDE GLOBAL SUMMARY]\n"
        f"{slide_summary.strip()}\n\n"
        "[LOCAL EVIDENCE]\n"
    )
    return prefix + raw_text


# =========================================================
# Memory bank
# =========================================================
class FileLockedJSONStore:
    def __init__(self, path):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            ensure_dir(parent)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"items": []}, f, ensure_ascii=False, indent=2)

    def read(self):
        with open(self.path, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        if "items" not in data:
            data = {"items": []}
        return data

    def update(self, update_fn):
        with open(self.path, "r+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    data = json.load(f)
                except Exception:
                    data = {"items": []}
                if "items" not in data:
                    data = {"items": []}

                new_data = update_fn(data)

                f.seek(0)
                f.truncate()
                json.dump(new_data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class MemoryBank:


    def __init__(self, path, max_items=50000):
        self.store = FileLockedJSONStore(path)
        self.path = path
        self.max_items = max_items

    def _iter_items(self):
        data = self.store.read()
        return data.get("items", [])

    def retrieve(self, query_emb, topk_phase1=12, topk_phase2=3, sim_threshold=0.15, lambda_q=0.5):
        data = self.store.read()
        items = data.get("items", [])
        if len(items) == 0:
            return []

        valid = []
        for item in items:
            emb = item.get("intent_emb", None)
            if emb is None:
                continue
            try:
                emb = np.asarray(emb, dtype=np.float32)
                emb = emb.reshape(1, -1)
                emb = safe_l2_normalize(emb, axis=-1)
                sim = float(np.dot(query_emb, emb.T).squeeze())
                if sim >= sim_threshold:
                    valid.append((item, sim))
            except Exception:
                continue

        if len(valid) == 0:
            return []

        valid = sorted(valid, key=lambda x: x[1], reverse=True)[:topk_phase1]

        sims = np.asarray([x[1] for x in valid], dtype=np.float32)
        qs = np.asarray([float(x[0].get("q_value", 0.0)) for x in valid], dtype=np.float32)

        sims_z = zscore(sims)
        qs_z = zscore(qs)

        scored = []
        for idx, (item, sim) in enumerate(valid):
            score = (1.0 - lambda_q) * sims_z[idx] + lambda_q * qs_z[idx]
            item_copy = deepcopy(item)
            item_copy["_retrieval_sim"] = float(sim)
            item_copy["_retrieval_score"] = float(score)
            scored.append(item_copy)

        scored = sorted(scored, key=lambda x: x["_retrieval_score"], reverse=True)[:topk_phase2]
        return scored

    def update_utilities(self, used_memory_ids, reward, alpha=0.3):
        if not used_memory_ids:
            return

        used_set = set(used_memory_ids)

        def _update(data):
            items = data.get("items", [])
            for item in items:
                if item.get("memory_id") in used_set:
                    old_q = float(item.get("q_value", 0.0))
                    new_q = old_q + alpha * (reward - old_q)
                    item["q_value"] = float(new_q)
                    item["n_used"] = int(item.get("n_used", 0)) + 1
                    if reward >= 0.5:
                        item["n_success"] = int(item.get("n_success", 0)) + 1
                    item["last_reward"] = float(reward)
                    item["last_updated_ts"] = time.time()
            data["items"] = items
            return data

        self.store.update(_update)

    def add_memory(self, memory_item):
        def _update(data):
            items = data.get("items", [])
            items.append(memory_item)
            if len(items) > self.max_items:
                items = items[-self.max_items:]
            data["items"] = items
            return data

        self.store.update(_update)

    def size(self):
        data = self.store.read()
        return len(data.get("items", []))


# =========================================================
# Memory helpers
# =========================================================
def build_intent_text(question_text, question_choices, task_type, slide_summary="", use_global_summary=False):
    choices_text = "; ".join(question_choices) if isinstance(question_choices, list) else str(question_choices)
    text = (
        f"Question: {question_text}\n"
        f"Choices: {choices_text}\n"
        f"TaskType: {task_type}\n"
    )
    if use_global_summary and slide_summary.strip():
        text += f"WholeSlideSummary: {truncate_text(slide_summary, 1200)}\n"
    return text.strip()


def encode_intent(plip, intent_text):
    emb = plip.encode_text([intent_text], batch_size=1)
    emb = safe_l2_normalize(emb, axis=-1)
    return emb


def build_experience_text(
    question_text,
    question_choices,
    task_type,
    slide_summary,
    accumulated_patch_names,
    all_results,
    final_answer,
    correct_answer,
):
    pred = final_answer["answer"]
    is_correct = str(pred).strip().lower() == str(correct_answer).strip().lower()

    if len(all_results) > 0:
        last_step = all_results[-1]
        last_mode = last_step.get("mode", "unknown")
        eval_result = last_step.get("evaluation_result", {})
        missing_info = eval_result.get("missing_info", "")
        zoom_reason = eval_result.get("zoom_reason", "")
    else:
        last_mode = "unknown"
        missing_info = ""
        zoom_reason = ""

    patch_preview = ", ".join(accumulated_patch_names[:12])

    if is_correct:
        text = (
            f"[SUCCESS CASE]\n"
            f"TaskType: {task_type}\n"
            f"Question: {question_text}\n"
            f"Choices: {'; '.join(question_choices)}\n"
            f"WholeSlideSummary: {truncate_text(slide_summary, 1200)}\n"
            f"EffectivePatchStrategy: Prioritize patches consistent with the slide-level lesion pattern; "
            f"cross-check local evidence with whole-slide context; if nuclear-level evidence is required, zoom into "
            f"high-cellularity / lesion-dominant regions.\n"
            f"ObservedWorkflow: {last_mode}\n"
            f"UsefulPatches: {patch_preview}\n"
            f"Answer: {pred}\n"
            f"Rationale: {truncate_text(final_answer.get('explanation', ''), 1200)}\n"
        )
    else:
        text = (
            f"[FAILURE REFLECTION]\n"
            f"TaskType: {task_type}\n"
            f"Question: {question_text}\n"
            f"Choices: {'; '.join(question_choices)}\n"
            f"WholeSlideSummary: {truncate_text(slide_summary, 1200)}\n"
            f"ObservedWorkflow: {last_mode}\n"
            f"LikelyErrorSource: The selected patches and reasoning path were insufficient or off-target.\n"
            f"MissingInfo: {truncate_text(str(missing_info), 400)}\n"
            f"ZoomReason: {truncate_text(str(zoom_reason), 400)}\n"
            f"UsedPatches: {patch_preview}\n"
            f"PredAnswer: {pred}\n"
            f"GroundTruth: {correct_answer}\n"
            f"CorrectionHint: Re-rank toward lesion-dominant, morphology-discriminative patches and enforce zoom when "
            f"nuclear / mitotic evidence is critical.\n"
        )

    return text.strip()


def format_memory_context(memories, max_chars=3000):
    if not memories:
        return ""

    blocks = []
    total = 0
    for i, m in enumerate(memories, start=1):
        block = (
            f"[MEMORY {i}]\n"
            f"TaskType: {m.get('task_type', '')}\n"
            f"Utility: {m.get('q_value', 0.0):.4f}\n"
            f"Experience: {m.get('experience_text', '')}\n"
        )
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    if not blocks:
        return ""

    return "[RETRIEVED MEMORY GUIDANCE]\n" + "\n".join(blocks)


def build_memory_guided_search_hint(memories):
    if not memories:
        return ""
    hints = []
    for m in memories:
        exp = m.get("experience_text", "")
        exp = truncate_text(exp, 600)
        hints.append(exp)
    return "\n".join(hints)


def compute_reward(pred_answer, correct_answer, reward_correct=1.0, reward_wrong=0.0):
    return float(reward_correct if str(pred_answer).strip().lower() == str(correct_answer).strip().lower() else reward_wrong)


# =========================================================
# Hybrid retrieval
# =========================================================
def build_multiview_queries(question_text, question_choices, slide_summary, task_type, memory_hint=""):
    choices_text = "; ".join(question_choices) if isinstance(question_choices, list) else str(question_choices)

    query_1 = question_text
    query_2 = f"Question: {question_text}\nChoices: {choices_text}"
    query_3 = (
        f"Task type: {task_type}. "
        f"For this pathology task, prioritize the most discriminative evidence relevant to the question. "
        f"Question: {question_text}. Choices: {choices_text}."
    )
    query_4 = (
        f"Whole-slide pathology summary: {slide_summary}\n"
        f"Current question: {question_text}\n"
        f"Choices: {choices_text}\n"
        f"Retrieve patches most likely to contain decisive evidence."
    )

    queries = [query_1, query_2, query_3]
    if slide_summary.strip():
        queries.append(query_4)
    if memory_hint.strip():
        queries.append(
            f"Retrieved experience guidance:\n{truncate_text(memory_hint, 1500)}\n"
            f"Question: {question_text}\nChoices: {choices_text}"
        )
    return queries


def encode_multiview_text_query(plip, queries):
    text_embs = plip.encode_text(queries, batch_size=min(4, len(queries)))
    text_embs = safe_l2_normalize(text_embs, axis=-1)
    merged = text_embs.mean(axis=0, keepdims=True)
    merged = safe_l2_normalize(merged, axis=-1)
    return merged


def compute_desc_relevance_scores(
    patch_names,
    descriptions_dict,
    question_text,
    question_choices,
    slide_summary,
    task_type,
    memory_hint="",
    max_keywords=64
):
    choices_text = "; ".join(question_choices) if isinstance(question_choices, list) else str(question_choices)

    full_query_text = (
        f"{question_text}\n"
        f"{choices_text}\n"
        f"{task_type}\n"
        f"{slide_summary}\n"
        f"{memory_hint}"
    )

    keywords = extract_keywords(full_query_text, max_keywords=max_keywords)
    keyword_set = set(keywords)

    pathology_focus = {
        "tumor", "tumour", "nuclear", "atypia", "mitosis", "mitotic", "necrosis",
        "gland", "stroma", "stromal", "invasion", "fibrosis", "lymphocyte",
        "inflammatory", "pleomorphism", "epithelial", "ductal", "lobular",
        "cellular", "hypercellular", "lesion", "malignant", "benign"
    }

    scores = []
    for p in patch_names:
        desc = descriptions_dict.get(p, "")
        desc_tokens = simple_tokenize(desc)
        if len(desc_tokens) == 0:
            scores.append(0.0)
            continue

        desc_set = set(desc_tokens)
        overlap = len(desc_set & keyword_set)
        pathology_overlap = len(desc_set & pathology_focus)
        richness = min(len(desc_tokens) / 50.0, 1.0)
        score = overlap + 0.3 * pathology_overlap + 0.5 * richness
        scores.append(float(score))

    return np.asarray(scores, dtype=np.float32)


def hybrid_rank_patches(
    feature_cache,
    patch_names,
    descriptions_dict,
    plip,
    question_text,
    question_choices,
    slide_summary,
    task_type,
    memory_hint="",
    visual_weight=0.75,
    desc_weight=0.25,
    max_keywords=64,
):
    if len(patch_names) == 0:
        return np.array([], dtype=np.float32)

    queries = build_multiview_queries(
        question_text=question_text,
        question_choices=question_choices,
        slide_summary=slide_summary,
        task_type=task_type,
        memory_hint=memory_hint
    )
    text_emb = encode_multiview_text_query(plip, queries)

    feats = np.stack([feature_cache[p] for p in patch_names], axis=0)
    feats = safe_l2_normalize(feats, axis=-1)
    visual_scores = (feats @ text_emb.T).squeeze()

    desc_scores = compute_desc_relevance_scores(
        patch_names=patch_names,
        descriptions_dict=descriptions_dict,
        question_text=question_text,
        question_choices=question_choices,
        slide_summary=slide_summary,
        task_type=task_type,
        memory_hint=memory_hint,
        max_keywords=max_keywords
    )

    visual_scores = zscore(visual_scores)
    desc_scores = zscore(desc_scores)

    final_scores = visual_weight * visual_scores + desc_weight * desc_scores
    return final_scores.astype(np.float32)


def hybrid_rank_for_missing_info(
    feature_cache,
    patch_names,
    descriptions_dict,
    plip,
    missing_info,
    slide_summary,
    question_text,
    question_choices,
    task_type,
    memory_hint="",
    visual_weight=0.75,
    desc_weight=0.25,
    max_keywords=64,
):
    if len(patch_names) == 0:
        return np.array([], dtype=np.float32)

    choices_text = "; ".join(question_choices) if isinstance(question_choices, list) else str(question_choices)

    queries = [
        missing_info,
        f"Question: {question_text}\nChoices: {choices_text}\nMissing evidence: {missing_info}",
        f"Task type: {task_type}\nWhole-slide summary: {slide_summary}\nMissing evidence to locate: {missing_info}",
    ]
    if memory_hint.strip():
        queries.append(
            f"Retrieved experience guidance:\n{truncate_text(memory_hint, 1500)}\n"
            f"Missing evidence: {missing_info}\nQuestion: {question_text}\nChoices: {choices_text}"
        )

    text_emb = encode_multiview_text_query(plip, queries)

    feats = np.stack([feature_cache[p] for p in patch_names], axis=0)
    feats = safe_l2_normalize(feats, axis=-1)
    visual_scores = (feats @ text_emb.T).squeeze()

    full_query_text = (
        f"{question_text}\n"
        f"{choices_text}\n"
        f"{task_type}\n"
        f"{slide_summary}\n"
        f"{missing_info}\n"
        f"{memory_hint}"
    )
    keywords = extract_keywords(full_query_text, max_keywords=max_keywords)
    keyword_set = set(keywords)

    desc_scores = []
    for p in patch_names:
        desc = descriptions_dict.get(p, "")
        desc_tokens = set(simple_tokenize(desc))
        overlap = len(desc_tokens & keyword_set)
        richness = min(len(desc_tokens) / 50.0, 1.0)
        desc_scores.append(float(overlap + 0.5 * richness))

    desc_scores = np.asarray(desc_scores, dtype=np.float32)

    visual_scores = zscore(visual_scores)
    desc_scores = zscore(desc_scores)

    final_scores = visual_weight * visual_scores + desc_weight * desc_scores
    return final_scores.astype(np.float32)


def resolve_long_id(target_long_id, all_descriptions):
    keys = list(all_descriptions.keys())

    if target_long_id in all_descriptions:
        return target_long_id, "exact"

    start_matches = [k for k in keys if k.startswith(target_long_id)]
    if start_matches:
        return start_matches[0], "key_startswith_target"

    reverse_start_matches = [k for k in keys if target_long_id.startswith(k)]
    if reverse_start_matches:
        return reverse_start_matches[0], "target_startswith_key"

    target_core = target_long_id.split(".")[0]
    core_matches = [k for k in keys if k.split(".")[0] == target_core]
    if core_matches:
        return core_matches[0], "core_id_match"

    return target_long_id, None


# =========================================================
# Models
# =========================================================
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


# =========================================================
# Main case process
# =========================================================
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
    memory_bank=None,
):
    target_long_id = pair["long_id"]

    resolved_long_id, match_mode = resolve_long_id(str(target_long_id), all_descriptions)
    if match_mode is not None and resolved_long_id != target_long_id:
        print(f"[GPU {rank}] Resolved long_id by {match_mode}: '{target_long_id}' -> '{resolved_long_id}'")
    elif match_mode is None:
        print(f"[GPU {rank}] Target ID '{target_long_id}' not found in description index.")

    target_long_id = resolved_long_id

    question_text = pair["question"]
    question_choices = pair["choices"]
    correct_answer = pair["answer"]
    #if float in questions_choices, convert to str 
    
    question_choices = [str(c) for c in question_choices]
    task_type = infer_task_type(question_text, question_choices)

    unique_id = make_unique_id(target_long_id, question_text)
    save_path = os.path.join(args.save_dir, f"{unique_id}.json")

    if os.path.exists(save_path):
        print(f"[GPU {rank}] Skipping: {unique_id} (Already completed)")
        return None

    print(f"\n[GPU {rank}] Starting Case: {unique_id}")
    print("=" * 100)
    print(f"[GPU {rank}] Question: {question_text}")
    print(f"[GPU {rank}] Choices: {question_choices}")
    print(f"[GPU {rank}] Ground Truth: {correct_answer}")
    print(f"[GPU {rank}] Task Type: {task_type}")

    descriptions_dict = all_descriptions.get(target_long_id)
    if not isinstance(descriptions_dict, dict) or len(descriptions_dict) == 0:
        print(f"[GPU {rank}] Missing or invalid descriptions for '{target_long_id}', skipping case.")
        return None

    descriptions_dict = deepcopy(descriptions_dict)
    print(f"[GPU {rank}] Extracted {len(descriptions_dict)} patch descriptions.")

    # 1. slide global summary
    global_summary_cache_dir = args.global_summary_cache_dir
    if global_summary_cache_dir is None:
        global_summary_cache_dir = os.path.join(args.save_dir, "_global_summaries")
    ensure_dir(global_summary_cache_dir)

    slide_global_summary = build_slide_level_global_summary(
        model=model,
        tokenizer=tokenizer,
        descriptions_dict=descriptions_dict,
        target_long_id=target_long_id,
        cache_dir=global_summary_cache_dir,
        chunk_size=args.global_summary_chunk_size,
        threshold=args.global_summary_threshold,
    )
    print(f"[GPU {rank}] Global summary built. Length={len(slide_global_summary)}")

    # 2. feature cache
    feature_cache = {}
    all_patch_names = list(descriptions_dict.keys())
    for patch_name in all_patch_names:
        npy_path = os.path.join(args.feature_dir, target_long_id, f"{patch_name.split('.')[0]}.npy")
        if os.path.exists(npy_path):
            feature_cache[patch_name] = np.load(npy_path)

    print(f"[GPU {rank}] Successfully cached {len(feature_cache)} / {len(all_patch_names)} patch features")

    available_patches = [p for p in all_patch_names if p in feature_cache]
    if not available_patches:
        print(f"[GPU {rank}] No available patches with features found. Skipping case.")
        return None

    # 3. memory retrieval before case inference
    retrieved_memories = []
    memory_context = ""
    memory_hint = ""
    used_memory_ids = []

    if args.memory_enabled and memory_bank is not None:
        intent_text = build_intent_text(
            question_text=question_text,
            question_choices=question_choices,
            task_type=task_type,
            slide_summary=slide_global_summary,
            use_global_summary=args.memory_use_global_summary_in_intent
        )
        intent_emb = encode_intent(plip, intent_text)

        retrieved_memories = memory_bank.retrieve(
            query_emb=intent_emb,
            topk_phase1=args.memory_topk_phase1,
            topk_phase2=args.memory_topk_phase2,
            sim_threshold=args.memory_sim_threshold,
            lambda_q=args.memory_lambda
        )
        used_memory_ids = [m["memory_id"] for m in retrieved_memories]
        memory_context = format_memory_context(retrieved_memories, max_chars=args.memory_max_context_chars)
        memory_hint = build_memory_guided_search_hint(retrieved_memories)

        print(f"[GPU {rank}] Retrieved {len(retrieved_memories)} memories.")
        for m in retrieved_memories:
            print(
                f"[GPU {rank}] Memory -> id={m.get('memory_id')} "
                f"sim={m.get('_retrieval_sim', 0):.4f} "
                f"score={m.get('_retrieval_score', 0):.4f} "
                f"Q={m.get('q_value', 0):.4f}"
            )

    # 4. initial patch selection: hybrid + memory hint
    initial_scores = hybrid_rank_patches(
        feature_cache=feature_cache,
        patch_names=available_patches,
        descriptions_dict=descriptions_dict,
        plip=plip,
        question_text=question_text,
        question_choices=question_choices,
        slide_summary=slide_global_summary,
        task_type=task_type,
        memory_hint=memory_hint,
        visual_weight=args.visual_weight,
        desc_weight=args.desc_weight,
        max_keywords=args.max_keywords,
    )

    K = max(1, int(len(all_patch_names) * args.initial_sample_ratio))
    topk_idx = np.argsort(initial_scores)[-K:][::-1]
    initial_sample_names = [available_patches[i] for i in topk_idx]

    accumulated_patch_names = list(initial_sample_names)
    patches_to_evaluate_names = list(initial_sample_names)
    remaining_patch_names = [p for p in all_patch_names if p not in accumulated_patch_names]

    print(f"[GPU {rank}] Initial sample size: {len(initial_sample_names)}; Remaining: {len(remaining_patch_names)}")
    print(f"[GPU {rank}] Initial selected patches (first 10): {initial_sample_names[:10]}")

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

        num_top_patches_to_describe = min(args.num_question_specific_desc, len(patches_to_evaluate_names))
        for i in range(num_top_patches_to_describe):
            patch_name = patches_to_evaluate_names[i]
            patch_path = get_patch_fullpath(args.patch_root, target_long_id, patch_name)
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
        desc_for_evaluation = prepend_global_summary_to_text(slide_global_summary, desc_for_evaluation)
        if memory_context:
            desc_for_evaluation = memory_context + "\n\n" + desc_for_evaluation

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
            final_desc_text = prepend_global_summary_to_text(slide_global_summary, final_desc_text)
            if memory_context:
                final_desc_text = memory_context + "\n\n" + final_desc_text

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
                "used_memory_ids": used_memory_ids,
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
            final_desc_text = prepend_global_summary_to_text(slide_global_summary, final_desc_text)
            if memory_context:
                final_desc_text = memory_context + "\n\n" + final_desc_text

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
                patch_feats = safe_l2_normalize(patch_feats, axis=-1)

                zoom_queries = build_multiview_queries(
                    question_text=question_text,
                    question_choices=question_choices,
                    slide_summary=slide_global_summary,
                    task_type=task_type,
                    memory_hint=memory_hint
                )
                text_emb = encode_multiview_text_query(plip, zoom_queries)
                sims = np.dot(patch_feats, text_emb.T).squeeze()

                top2_idx = np.argsort(sims)[-2:][::-1]
                top2_names = [patch_names_valid[i] for i in top2_idx]
                print(f"[GPU {rank}] -> Selected Top2 patches for zoom: {top2_names}")

                candidate_images = []
                candidate_names = []
                for patch_name in top2_names:
                    patch_path = get_patch_fullpath(args.patch_root, target_long_id, patch_name)
                    sub_patches = split_patch_for_zoom(patch_path, zoom_level_val)
                    for sub_img, (x, y) in sub_patches:
                        candidate_images.append(sub_img)
                        candidate_names.append(f"{x}_{y}")

                if not candidate_images:
                    print(f"[GPU {rank}] No sub-patches generated, skipping zoom process.")
                else:
                    image_embs = plip.encode_images(candidate_images, batch_size=4)
                    image_embs = safe_l2_normalize(image_embs, axis=-1)
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
                        question_text,
                        question_choices,
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
                        "used_memory_ids": used_memory_ids,
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

            replenish_scores = hybrid_rank_for_missing_info(
                feature_cache=feature_cache,
                patch_names=valid_remaining_names,
                descriptions_dict=descriptions_dict,
                plip=plip,
                missing_info=missing_info,
                slide_summary=slide_global_summary,
                question_text=question_text,
                question_choices=question_choices,
                task_type=task_type,
                memory_hint=memory_hint,
                visual_weight=args.visual_weight,
                desc_weight=args.desc_weight,
                max_keywords=args.max_keywords,
            )

            num_to_add = max(1, int(len(all_patch_names) * args.replenish_ratio))
            best_indices = np.argsort(replenish_scores)[::-1][:num_to_add]
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
        final_desc_text = prepend_global_summary_to_text(slide_global_summary, final_desc_text)
        if memory_context:
            final_desc_text = memory_context + "\n\n" + final_desc_text

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
            "used_memory_ids": used_memory_ids,
            "answer": final_answer["answer"],
            "explanation": final_answer["explanation"]
        })

    print(f"[GPU {rank}] single_case_result: {final_answer['answer']}")
    print(f"[GPU {rank}] ground_truth: {correct_answer}")

    # 5. memory update/writeback only in build_memory mode
    reward = compute_reward(
        pred_answer=final_answer["answer"],
        correct_answer=correct_answer,
        reward_correct=args.reward_correct,
        reward_wrong=args.reward_wrong
    )

    if args.mode == "build_memory" and args.memory_enabled and memory_bank is not None:
        if used_memory_ids:
            memory_bank.update_utilities(
                used_memory_ids=used_memory_ids,
                reward=reward,
                alpha=args.memory_update_alpha
            )

        new_intent_text = build_intent_text(
            question_text=question_text,
            question_choices=question_choices,
            task_type=task_type,
            slide_summary=slide_global_summary,
            use_global_summary=args.memory_use_global_summary_in_intent
        )
        new_intent_emb = encode_intent(plip, new_intent_text).squeeze().astype(np.float32).tolist()
        experience_text = build_experience_text(
            question_text=question_text,
            question_choices=question_choices,
            task_type=task_type,
            slide_summary=slide_global_summary,
            accumulated_patch_names=accumulated_patch_names,
            all_results=all_results,
            final_answer=final_answer,
            correct_answer=correct_answer,
        )

        memory_item = {
            "memory_id": str(uuid.uuid4()),
            "long_id": target_long_id,
            "task_type": task_type,
            "intent_text": new_intent_text,
            "intent_emb": new_intent_emb,
            "experience_text": experience_text,
            "q_value": float(reward),
            "n_used": 0,
            "n_success": 0,
            "last_reward": float(reward),
            "mode_source": args.mode,
            "created_ts": time.time(),
            "meta": {
                "dataset_name": args.dataset_name,
                "question": question_text,
                "choices": question_choices,
                "ground_truth": correct_answer,
                "pred_answer": final_answer["answer"]
            }
        }
        memory_bank.add_memory(memory_item)

    case_result = {
        "long_id": target_long_id,
        "question": question_text,
        "choices": question_choices,
        "ground_truth": correct_answer,
        "pred_answer": final_answer["answer"],
        "reward": reward,
        "task_type": task_type,
        "slide_global_summary": slide_global_summary,
        "used_memory_ids": used_memory_ids,
        "memory_retrieved": [
            {
                "memory_id": m.get("memory_id"),
                "q_value": m.get("q_value"),
                "retrieval_sim": m.get("_retrieval_sim"),
                "retrieval_score": m.get("_retrieval_score"),
                "task_type": m.get("task_type")
            } for m in retrieved_memories
        ],
        "explanation": final_answer["explanation"],
        "process": all_results
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(case_result, f, indent=2, ensure_ascii=False)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return case_result


# =========================================================
# Worker / merge
# =========================================================
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

    memory_bank = None
    if args.memory_enabled:
        memory_bank = MemoryBank(args.memory_bank_path, max_items=args.memory_max_items)
        print(f"[Worker {rank}] Memory bank enabled: {args.memory_bank_path} (size={memory_bank.size()})")

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
                memory_bank=memory_bank,
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

    if memory_bank is not None:
        print(f"[Worker {rank}] Final memory bank size: {memory_bank.size()}")

    print(f"[Worker {rank}] Finished. Saved worker summary to {summary_path}")


def merge_results(save_dir):
    all_case_results = []
    for f in os.listdir(save_dir):
        if f.endswith(".json") and not f.startswith("_worker_") and f != "all_results.json":
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
    print(json.dumps(all_case_results[:3], indent=2, ensure_ascii=False))
    print(f"\n[Merge] Total cases merged: {len(all_case_results)}")
    print(f"[Merge] Final merged file saved to: {final_path}")


# =========================================================
# main
# =========================================================
def main():
    os.makedirs(args.save_dir, exist_ok=True)

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip() != ""]
    if len(gpu_ids) == 0:
        raise ValueError("No valid gpu_ids provided.")
    if args.procs_per_gpu < 1:
        raise ValueError("procs_per_gpu must be >= 1.")

    worker_gpu_ids = []
    for gid in gpu_ids:
        worker_gpu_ids.extend([gid] * args.procs_per_gpu)
    total_workers = len(worker_gpu_ids)

    if args.global_summary_cache_dir is None:
        args.global_summary_cache_dir = os.path.join(args.save_dir, "_global_summaries")

    if args.memory_bank_path is None:
        args.memory_bank_path = os.path.join(args.save_dir, "memory_bank.json")

    print("=" * 100)
    print(f"Mode:                  {args.mode}")
    print(f"PLIP Lib:              {args.plip_lib_path}")
    print(f"Model Qwen:            {args.qwen_ckpt}")
    print(f"Model PLIP:            {args.plip_ckpt}")
    print(f"Model PR1:             {args.patho_r1_ckpt}")
    print(f"Dataset:               {args.dataset_name}")
    print(f"Questions:             {args.questions_file}")
    print(f"Results to:            {args.save_dir}")
    print(f"Global summary cache:  {args.global_summary_cache_dir}")
    print(f"Memory enabled:        {args.memory_enabled}")
    print(f"Memory bank path:      {args.memory_bank_path}")
    print(f"Memory lambda:         {args.memory_lambda}")
    print(f"Memory topk p1/p2:     {args.memory_topk_phase1}/{args.memory_topk_phase2}")
    print(f"Memory sim threshold:  {args.memory_sim_threshold}")
    print(f"Memory update alpha:   {args.memory_update_alpha}")
    print(f"GPU IDs:               {gpu_ids}")
    print(f"Proc/GPU:              {args.procs_per_gpu}")
    print(f"Workers:               {total_workers}")
    print(f"Visual weight:         {args.visual_weight}")
    print(f"Desc weight:           {args.desc_weight}")
    print("=" * 100)

    mp.set_start_method("spawn", force=True)
    mp.spawn(worker_main, args=(worker_gpu_ids, args), nprocs=total_workers, join=True)

    merge_results(args.save_dir)


if __name__ == "__main__":
    main()





# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pathagent.py \
#     --mode build_memory \
#     --memory_enabled \
#     --gpu_ids 0,1,2,3,4,5,6,7 \
#     --plip_lib_path ./plip \
#     --qwen_ckpt /data/yuhaowang/cache/qwen3b/ \
#     --procs_per_gpu 2 \
#     --plip_ckpt /data/yuhaowang/cache/plip \
#     --patho_r1_ckpt /data/yuhaowang/cache/Patho-R1-7B \
#     --descriptions_file /data/yuhaowang/processed_wsi/TCGA-BRCA/desc/patches_descriptions.json \
#     --questions_file /data/yuhaowang/wsi-vqa/WsiVQA_train.json \
#     --feature_dir /data/yuhaowang/processed_wsi/TCGA-BRCA/img_features \
#     --patch_root /data/yuhaowang/processed_wsi/TCGA-BRCA/patches_output \
#     --save_dir /data/yuhaowang/processed_wsi/TCGA-BRCA/results/pathagent_memory_train \
#     --memory_bank_path /data/yuhaowang/processed_wsi/TCGA-BRCA/results/pathagent_memory_train/memory_bank.json \
#     --visual_weight 0.75 \
#     --desc_weight 0.25 \
#     --memory_lambda 0.5 \
#     --memory_topk_phase1 12 \
#     --memory_topk_phase2 3 \
#     --memory_sim_threshold 0.15 \
#     --memory_update_alpha 0.3



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pathagent.py \
#     --mode eval \
#     --memory_enabled \
#     --gpu_ids 0,1,2,3,4,5,6,7 \
#     --plip_lib_path ./plip \
#     --qwen_ckpt /data/yuhaowang/cache/qwen3b/ \
#     --procs_per_gpu 2 \
#     --plip_ckpt /data/yuhaowang/cache/plip \
#     --patho_r1_ckpt /data/yuhaowang/cache/Patho-R1-7B/ \
#     --descriptions_file /data/yuhaowang/processed_wsi/TCGA-BRCA/desc/patches_descriptions.json \
#     --questions_file /data/yuhaowang/wsi-vqa/WsiVQA_test.json \
#     --feature_dir /data/yuhaowang/processed_wsi/TCGA-BRCA/img_features \
#     --patch_root /data/yuhaowang/processed_wsi/TCGA-BRCA/patches_output \
#     --save_dir /data/yuhaowang/processed_wsi/TCGA-BRCA/results/pathagent_memory_test \
#     --memory_bank_path /data/yuhaowang/processed_wsi/TCGA-BRCA/results/pathagent_memory_train/memory_bank.json \
#     --visual_weight 0.75 \
#     --desc_weight 0.25 \
#     --memory_lambda 0.5 \
#     --memory_topk_phase1 12 \
#     --memory_topk_phase2 3 \
#     --memory_sim_threshold 0.15




# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pathagent.py \
#     --mode build_memory \
#     --memory_enabled \
#     --gpu_ids 0,1,2,3,4,5,6,7 \
#     --plip_lib_path ./plip \
#     --qwen_ckpt /data/yuhaowang/cache/qwen3b/ \
#     --procs_per_gpu 2 \
#     --plip_ckpt /data/yuhaowang/cache/plip \
#     --patho_r1_ckpt /data/yuhaowang/cache/Patho-R1-7B \
#     --descriptions_file /data2/yuhaowang/processed_wsi/slidebench/desc/patches_descriptions.json \
#     --questions_file /data2/yuhaowang/processed_wsi/slidebench/vqa/SlideBench-VQA.csv \
#     --feature_dir /data2/yuhaowang/processed_wsi/slidebench/img_features \
#     --patch_root /data2/yuhaowang/processed_wsi/slidebench/patches_output \
#     --save_dir /data2/yuhaowang/processed_wsi/slidebench/results/pathagent_memory_train \
#     --memory_bank_path /data2/yuhaowang/processed_wsi/slidebench/results/pathagent_memory_train/memory_bank.json \
#     --visual_weight 0.75 \
#     --dataset_name slidebench_vqa \
#     --desc_weight 0.25 \
#     --memory_lambda 0.5 \
#     --memory_topk_phase1 12 \
#     --memory_topk_phase2 3 \
#     --memory_sim_threshold 0.15 \
#     --memory_update_alpha 0.3