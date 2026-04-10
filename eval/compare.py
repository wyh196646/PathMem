import os
import json
import hashlib
import difflib
import argparse
from typing import Dict, Any, Optional, List


def make_unique_id(long_id, question_text):
    q_hash = hashlib.md5(question_text.encode("utf-8")).hexdigest()[:8]
    return f"{long_id}_{q_hash}"


def acc_of_seq(choices, gt, res):
    """
    Determine if prediction 'res' is most similar to 'gt' among choices.
    Returns:
        True / False / None
    """
    if not choices:
        return None

    gt = (gt or "").strip()
    res = (res or "").strip()

    score = difflib.SequenceMatcher(None, res, gt).quick_ratio()
    for item in choices:
        tmp = difflib.SequenceMatcher(None, res, str(item)).quick_ratio()
        if tmp > score:
            return False
    return True


def normalize_text(x):
    if x is None:
        return ""
    return str(x).strip()


def load_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all json files in a directory and index by unique case id.
    """
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory does not exist: {results_dir}")

    data_dict = {}
    file_list = sorted([f for f in os.listdir(results_dir) if f.endswith(".json")])

    for fname in file_list:
        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to read {fpath}: {e}")
            continue

        long_id = data.get("long_id")
        question = data.get("question", "")

        if not long_id or not question:
            print(f"[Warning] Skip invalid file (missing long_id/question): {fpath}")
            continue

        case_id = make_unique_id(long_id, question)

        gt = normalize_text(data.get("ground_truth"))
        pred = normalize_text(data.get("pred_answer"))
        choices = data.get("choices", None)

        # 判断是否答对
        is_correct = None
        if isinstance(choices, list) and len(choices) > 0:
            is_correct = acc_of_seq(choices, gt, pred)
        else:
            # 非 choices 类型的数据，这里不强行判断
            # 你也可以按需要改成字符串完全匹配：
            # is_correct = (pred == gt)
            is_correct = None

        data_dict[case_id] = {
            "case_id": case_id,
            "file_name": fname,
            "long_id": long_id,
            "question": question,
            "ground_truth": gt,
            "pred_answer": pred,
            "choices": choices,
            "explanation": data.get("explanation", ""),
            "raw_data": data,
            "is_correct": is_correct,
        }

    return data_dict


def compare_two_dirs(a_dir: str, b_dir: str):
    a_data = load_results(a_dir)
    b_data = load_results(b_dir)

    a_ids = set(a_data.keys())
    b_ids = set(b_data.keys())

    common_ids = sorted(a_ids & b_ids)
    only_a = sorted(a_ids - b_ids)
    only_b = sorted(b_ids - a_ids)

    a_correct_b_wrong = []
    b_correct_a_wrong = []
    both_correct = []
    both_wrong = []
    skipped = []

    for cid in common_ids:
        a_item = a_data[cid]
        b_item = b_data[cid]

        a_ok = a_item["is_correct"]
        b_ok = b_item["is_correct"]

        # 只比较那些能判断正确性的 case
        if a_ok is None or b_ok is None:
            skipped.append({
                "case_id": cid,
                "reason": "Cannot determine correctness for at least one side",
                "A_file": a_item["file_name"],
                "B_file": b_item["file_name"],
                "question": a_item["question"],
                "ground_truth": a_item["ground_truth"],
                "A_pred": a_item["pred_answer"],
                "B_pred": b_item["pred_answer"],
                "choices": a_item["choices"] if a_item["choices"] is not None else b_item["choices"],
            })
            continue

        record = {
            "case_id": cid,
            "long_id": a_item["long_id"],
            "question": a_item["question"],
            "ground_truth": a_item["ground_truth"],
            "choices": a_item["choices"] if a_item["choices"] is not None else b_item["choices"],

            "A_file": a_item["file_name"],
            "A_pred": a_item["pred_answer"],
            "A_correct": a_ok,
            "A_explanation": a_item["explanation"],

            "B_file": b_item["file_name"],
            "B_pred": b_item["pred_answer"],
            "B_correct": b_ok,
            "B_explanation": b_item["explanation"],
        }

        if a_ok is True and b_ok is False:
            a_correct_b_wrong.append(record)
        elif a_ok is False and b_ok is True:
            b_correct_a_wrong.append(record)
        elif a_ok is True and b_ok is True:
            both_correct.append(record)
        elif a_ok is False and b_ok is False:
            both_wrong.append(record)

    summary = {
        "A_total_files": len(a_data),
        "B_total_files": len(b_data),
        "common_cases": len(common_ids),
        "only_in_A": len(only_a),
        "only_in_B": len(only_b),
        "A_correct_B_wrong": len(a_correct_b_wrong),
        "B_correct_A_wrong": len(b_correct_a_wrong),
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "skipped": len(skipped),
    }

    results = {
        "summary": summary,
        "only_in_A_case_ids": only_a,
        "only_in_B_case_ids": only_b,
        "A_correct_B_wrong_cases": a_correct_b_wrong,
        "B_correct_A_wrong_cases": b_correct_a_wrong,
        "both_correct_cases": both_correct,
        "both_wrong_cases": both_wrong,
        "skipped_cases": skipped,
    }

    return results


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Compare results from two directories")
    parser.add_argument("--a_dir", type=str, required=True, help="Path to model A results directory")
    parser.add_argument("--b_dir", type=str, required=True, help="Path to model B results directory")
    parser.add_argument("--output_dir", type=str, default="./compare_output", help="Directory to save comparison results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = compare_two_dirs(args.a_dir, args.b_dir)

    # 保存总结果
    save_json(results, os.path.join(args.output_dir, "full_compare_results.json"))

    # 单独保存重点 case
    save_json(results["A_correct_B_wrong_cases"], os.path.join(args.output_dir, "A_correct_B_wrong.json"))
    save_json(results["B_correct_A_wrong_cases"], os.path.join(args.output_dir, "B_correct_A_wrong.json"))
    save_json(results["skipped_cases"], os.path.join(args.output_dir, "skipped_cases.json"))

    print("\n===== Compare Summary =====")
    for k, v in results["summary"].items():
        print(f"{k}: {v}")

    print("\nSaved files:")
    print(os.path.join(args.output_dir, "full_compare_results.json"))
    print(os.path.join(args.output_dir, "A_correct_B_wrong.json"))
    print(os.path.join(args.output_dir, "B_correct_A_wrong.json"))
    print(os.path.join(args.output_dir, "skipped_cases.json"))


if __name__ == "__main__":
    main()


# python compare.py \
#   --a_dir /data/yuhaowang/processed_wsi/TCGA-BRCA/results/pathagent_memory_test \
#   --b_dir /data/yuhaowang/processed_wsi/TCGA-BRCA/results/pathagent \
#   --output_dir ./