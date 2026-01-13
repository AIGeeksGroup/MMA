import json
import argparse
import re
from collections import Counter, defaultdict
from typing import Dict, Any, Tuple, List

MISSING_PATTERNS = [
    r"\bno information\b",
    r"\bno available information\b",
    r"\bnot enough info\b",
    r"\bno specific information\b",
    r"\bno explicit information\b",
    r"\bnot recorded\b",
    r"\bnot specified\b",
    r"\bunknown\b",
    r"\bcannot determine\b",
    r"\bunable to determine\b",
]

def is_missing_info(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    for pat in MISSING_PATTERNS:
        if re.search(pat, t):
            return True
    return False

def load_metrics(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("detailed_results", [])

def build_index(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for it in items:
        q = it.get("question")
        if isinstance(q, str):
            idx[q] = {
                "answer": it.get("answer"),
                "response": it.get("response"),
                "llm_score": it.get("llm_score"),
                "category": it.get("category"),
                "bleu_score": it.get("bleu_score"),
                "f1_score": it.get("f1_score"),
            }
    return idx

def compare(v2_idx: Dict[str, Dict[str, Any]], base_idx: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    # 选出 v2 中判定为缺失信息的题目
    v2_missing_questions = [q for q, it in v2_idx.items() if is_missing_info(it.get("response"))]
    summary_counts = Counter()
    per_question: List[Dict[str, Any]] = []

    for q in v2_missing_questions:
        v2_item = v2_idx.get(q, {})
        base_item = base_idx.get(q)

        base_label = "not_found"
        if base_item:
            if is_missing_info(base_item.get("response")):
                base_label = "baseline_missing_info"
            else:
                # 用 llm_score 大致判断正确/错误
                base_label = "baseline_correct" if int(base_item.get("llm_score") or 0) == 1 else "baseline_wrong"

        summary_counts[base_label] += 1
        per_question.append({
            "question": q,
            "v2_response": v2_item.get("response"),
            "v2_answer": v2_item.get("answer"),
            "v2_llm_score": v2_item.get("llm_score"),
            "baseline_response": base_item.get("response") if base_item else None,
            "baseline_answer": base_item.get("answer") if base_item else None,
            "baseline_llm_score": base_item.get("llm_score") if base_item else None,
            "baseline_label_for_this_compare": base_label,
            "category": v2_item.get("category"),
        })

    total = len(v2_missing_questions) or 1
    summary_pct = {k: round(v * 100.0 / total, 2) for k, v in summary_counts.items()}

    return {
        "total_v2_missing_info": len(v2_missing_questions),
        "summary_counts": dict(summary_counts),
        "summary_percentages": summary_pct,
        "examples": per_question,
    }

def main():
    parser = argparse.ArgumentParser(description="Compare baseline responses when v2 chooses missing info.")
    parser.add_argument("--v2_metrics", type=str, default="./results/v2/evaluation_metrics.json")
    parser.add_argument("--baseline_metrics", type=str, default="./results/mirix/evaluation_metrics.json")
    parser.add_argument("--output", type=str, default="./results/v2/missing_info_compare.json")
    parser.add_argument("--print_examples", type=int, default=20)
    args = parser.parse_args()

    v2_items = load_metrics(args.v2_metrics)
    base_items = load_metrics(args.baseline_metrics)

    v2_idx = build_index(v2_items)
    base_idx = build_index(base_items)

    result = compare(v2_idx, base_idx)

    # 输出摘要
    print("Total v2 missing-info questions:", result["total_v2_missing_info"])
    print("Summary:", result["summary_counts"])
    print("Percentages:", result["summary_percentages"])

    # 打印若干示例
    for ex in result["examples"][: args.print_examples]:
        print("- Q:", ex["question"])
        print("  v2:", ex["v2_response"])
        print("  baseline:", ex["baseline_response"])
        print("  baseline_label:", ex["baseline_label_for_this_compare"])

    # 写文件
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()