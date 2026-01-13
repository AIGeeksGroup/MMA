import argparse, json, re, math
from typing import List, Dict, Any
from collections import defaultdict

ABSTAIN_PATTERNS = [
    r"There is no information",
    r"no available information",
    r"NOT ENOUGH INFO",
    r"There is no specific information",
    r"No information available",
    r"There is no explicit information",
]

def load_entries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ["detailed_results", "results", "evaluations", "entries", "data"]:
            if k in data and isinstance(data[k], list):
                return data[k]
    # fallback: flatten values that look like lists
    entries = []
    for v in (data.values() if isinstance(data, dict) else []):
        if isinstance(v, list):
            entries.extend(v)
    return entries

def get_field(d: Dict[str, Any], candidates: List[str], default=None):
    for k in candidates:
        if k in d:
            return d[k]
    return default

def is_abstain(text: str, abstain_regex: re.Pattern) -> bool:
    if not isinstance(text, str):
        return False
    return bool(abstain_regex.search(text))

def is_correct(entry: Dict[str, Any], thr_llm: float, thr_f1: float, thr_bleu: float) -> bool:
    llm = get_field(entry, ["llm_score", "llmScore", "llm"], None)
    f1 = get_field(entry, ["f1_score", "f1"], None)
    bleu = get_field(entry, ["bleu_score", "bleu"], None)
    if isinstance(llm, (int, float)):
        return llm >= thr_llm
    if isinstance(f1, (int, float)):
        return f1 >= thr_f1
    if isinstance(bleu, (int, float)):
        return bleu >= thr_bleu
    # very conservative fallback: exact match (rarely used)
    ref = get_field(entry, ["correct_answer", "answer", "reference"], None)
    rsp = get_field(entry, ["model_response", "response", "model_output"], "")
    if isinstance(ref, str) and isinstance(rsp, str):
        return ref.strip().lower() == rsp.strip().lower()
    return False

def index_by_question(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for e in entries:
        q = get_field(e, ["question", "prompt", "query"], None)
        if isinstance(q, str):
            if q not in idx:
                idx[q] = e
    return idx

def run(v2_path: str, base_path: str, thr_llm: float, thr_f1: float, thr_bleu: float,
        lambda_penalty: float, print_examples: int, output: str):
    v2_entries = load_entries(v2_path)
    b_entries = load_entries(base_path)
    abstain_regex = re.compile("|".join(ABSTAIN_PATTERNS), re.IGNORECASE)
    v2_idx = index_by_question(v2_entries)
    b_idx = index_by_question(b_entries)

    # Subset：v2拒答的题目
    v2_abstain_questions = []
    for q, e in v2_idx.items():
        rsp = get_field(e, ["model_response", "response", "model_output"], "")
        if is_abstain(rsp, abstain_regex):
            v2_abstain_questions.append(q)

    subset_total = len(v2_abstain_questions)
    sub_correct = sub_wrong = sub_missing = 0
    examples = []
    for q in v2_abstain_questions:
        v2e = v2_idx.get(q)
        be = b_idx.get(q)
        v2_rsp = get_field(v2e, ["model_response", "response", "model_output"], "")
        if be is None:
            sub_missing += 1
            examples.append({"q": q, "v2": v2_rsp, "baseline": "<no entry>", "baseline_label": "baseline_missing"})
            continue
        b_rsp = get_field(be, ["model_response", "response", "model_output"], "")
        if is_abstain(b_rsp, abstain_regex):
            sub_missing += 1
            lbl = "baseline_missing_info"
        else:
            if is_correct(be, thr_llm, thr_f1, thr_bleu):
                sub_correct += 1
                lbl = "baseline_correct"
            else:
                sub_wrong += 1
                lbl = "baseline_wrong"
        if len(examples) < print_examples:
            examples.append({"q": q, "v2": v2_rsp, "baseline": b_rsp, "baseline_label": lbl})

    sub_answered = sub_correct + sub_wrong
    sub_wrong_rate = (sub_wrong / sub_answered) if sub_answered > 0 else 0.0
    sub_coverage = (sub_answered / subset_total) if subset_total > 0 else 0.0
    lambda_threshold = (sub_correct / sub_wrong) if sub_wrong > 0 else math.inf

    # Overall：两条跑全面统计
    def overall_stats(entries: List[Dict[str, Any]]):
        total = len(entries)
        answered = correct = wrong = 0
        sum_llm_answered = 0.0
        for e in entries:
            rsp = get_field(e, ["model_response", "response", "model_output"], "")
            if not is_abstain(rsp, abstain_regex):
                answered += 1
                if is_correct(e, thr_llm, thr_f1, thr_bleu):
                    correct += 1
                else:
                    wrong += 1
                llm = get_field(e, ["llm_score", "llmScore", "llm"], None)
                if isinstance(llm, (int, float)):
                    sum_llm_answered += llm
        coverage = answered / total if total > 0 else 0.0
        acc = correct / answered if answered > 0 else 0.0
        w_rate = wrong / answered if answered > 0 else 0.0
        avg_llm_answered = (sum_llm_answered / answered) if answered > 0 else None
        return {
            "total": total, "answered": answered, "coverage": round(coverage, 4),
            "correct": correct, "wrong": wrong, "accuracy_answered": round(acc, 4),
            "wrong_rate_answered": round(w_rate, 4),
            "avg_llm_among_answered": round(avg_llm_answered, 4) if avg_llm_answered is not None else None,
        }

    v2_overall = overall_stats(v2_entries)
    b_overall = overall_stats(b_entries)

    def utility(stats: Dict[str, Any], lam: float):
        return stats["correct"] - lam * stats["wrong"]

    summary = {
        "subset_v2_abstain_total": subset_total,
        "subset_baseline": {
            "correct": sub_correct, "wrong": sub_wrong, "missing_info": sub_missing,
            "answered": sub_answered, "coverage": round(sub_coverage, 4),
            "wrong_rate_answered": round(sub_wrong_rate, 4),
            "lambda_threshold_v2_beats_baseline_on_subset": round(lambda_threshold, 4),
        },
        "overall": {
            "v2": {**v2_overall, "utility_lambda": {str(lambda_penalty): utility(v2_overall, lambda_penalty)}},
            "baseline": {**b_overall, "utility_lambda": {str(lambda_penalty): utility(b_overall, lambda_penalty)}},
        },
        "examples": examples,
        "thresholds_used": {"llm": thr_llm, "f1": thr_f1, "bleu": thr_bleu, "lambda_penalty": lambda_penalty},
    }

    print(f"Subset (v2 abstain) total={subset_total} | baseline: "
          f"answered={sub_answered}, correct={sub_correct}, wrong={sub_wrong}, missing={sub_missing}, "
          f"coverage={sub_coverage:.3f}, wrong_rate={sub_wrong_rate:.3f}, "
          f"lambda_threshold={lambda_threshold:.3f}")
    print(f"Overall v2: {v2_overall} | utility@λ={lambda_penalty}: {summary['overall']['v2']['utility_lambda'][str(lambda_penalty)]}")
    print(f"Overall baseline: {b_overall} | utility@λ={lambda_penalty}: {summary['overall']['baseline']['utility_lambda'][str(lambda_penalty)]}")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_metrics", required=True)
    ap.add_argument("--baseline_metrics", required=True)
    ap.add_argument("--llm_threshold", type=float, default=0.75)
    ap.add_argument("--f1_threshold", type=float, default=0.5)
    ap.add_argument("--bleu_threshold", type=float, default=0.3)
    ap.add_argument("--lambda_penalty", type=float, default=2.0)
    ap.add_argument("--print_examples", type=int, default=0)
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()
    run(args.v2_metrics, args.baseline_metrics, args.llm_threshold, args.f1_threshold,
        args.bleu_threshold, args.lambda_penalty, args.print_examples, args.output)

if __name__ == "__main__":
    main()