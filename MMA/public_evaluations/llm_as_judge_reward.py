import argparse, json, re, math
from typing import List, Dict, Any

"""Compute utility for judge outputs with abstain reward and wrong penalty."""

ABSTAIN_PATTERNS = [
    r"There is no information",
    r"no available information",
    r"NOT ENOUGH INFO",
    r"There is no specific information",
    r"No information available",
    r"There is no explicit information",
    r"not specified",
    r"unknown",
    r"cannot determine",
    r"unable to determine",
]

def load_entries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "detailed_results" in data:
        return data["detailed_results"]
    if isinstance(data, list):
        return data
    return []

def is_abstain(text: Any, abstain_regex: re.Pattern) -> bool:
    if not isinstance(text, str):
        return False
    return bool(abstain_regex.search(text))

def get_field(d: Dict[str, Any], candidates: List[str], default=None):
    for k in candidates:
        if k in d:
            return d[k]
    return default

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
    # Fallback exact-match
    ref = get_field(entry, ["correct_answer", "answer", "reference"], None)
    rsp = get_field(entry, ["model_response", "response", "model_output"], "")
    if isinstance(ref, str) and isinstance(rsp, str):
        return ref.strip().lower() == rsp.strip().lower()
    return False

def overall_stats(entries: List[Dict[str, Any]], abstain_regex: re.Pattern,
                  thr_llm: float, thr_f1: float, thr_bleu: float) -> Dict[str, Any]:
    total = len(entries)
    answered = abstained = correct = wrong = 0
    sum_llm_answered = 0.0
    for e in entries:
        rsp = get_field(e, ["model_response", "response", "model_output"], "")
        if is_abstain(rsp, abstain_regex):
            abstained += 1
            continue
        answered += 1
        if is_correct(e, thr_llm, thr_f1, thr_bleu):
            correct += 1
        else:
            wrong += 1
        llm = get_field(e, ["llm_score", "llmScore", "llm"], None)
        if isinstance(llm, (int, float)):
            sum_llm_answered += llm

    coverage = answered / total if total > 0 else 0.0
    acc_answered = correct / answered if answered > 0 else 0.0
    wrong_rate_answered = wrong / answered if answered > 0 else 0.0
    avg_llm_answered = (sum_llm_answered / answered) if answered > 0 else None

    return {
        "total": total,
        "answered": answered,
        "abstained": abstained,
        "coverage": round(coverage, 4),
        "correct": correct,
        "wrong": wrong,
        "accuracy_answered": round(acc_answered, 4),
        "wrong_rate_answered": round(wrong_rate_answered, 4),
        "avg_llm_among_answered": round(avg_llm_answered, 4) if avg_llm_answered is not None else None,
    }

def compute_utility(stats: Dict[str, Any], lam_wrong: float, reward_abstain: float) -> float:
    # Utility: correct - λ*wrong + r*abstain
    return stats["correct"] - lam_wrong * stats["wrong"] + reward_abstain * stats["abstained"]

def run(input_path: str, thr_llm: float, thr_f1: float, thr_bleu: float,
        lam_wrong: float, reward_abstain: float, print_examples: int, output: str) -> Dict[str, Any]:
    entries = load_entries(input_path)
    abstain_regex = re.compile("|".join(ABSTAIN_PATTERNS), re.IGNORECASE)
    stats = overall_stats(entries, abstain_regex, thr_llm, thr_f1, thr_bleu)
    util = compute_utility(stats, lam_wrong, reward_abstain)

    examples = []
    cnt = 0
    for e in entries:
        rsp = get_field(e, ["model_response", "response", "model_output"], "")
        if is_abstain(rsp, abstain_regex):
            if cnt < print_examples:
                examples.append({
                    "question": get_field(e, ["question", "prompt", "query"], None),
                    "response": rsp,
                    "answer": get_field(e, ["answer", "reference", "correct_answer"], None),
                    "llm_score": get_field(e, ["llm_score", "llmScore", "llm"], None),
                    "f1_score": get_field(e, ["f1_score", "f1"], None),
                    "bleu_score": get_field(e, ["bleu_score", "bleu"], None),
                    "category": get_field(e, ["category"], None),
                })
                cnt += 1

    summary = {
        "thresholds": {"llm": thr_llm, "f1": thr_f1, "bleu": thr_bleu},
        "penalties": {"lambda_wrong": lam_wrong, "reward_abstain": reward_abstain},
        "overall_stats": stats,
        "utility": util,
        "examples_abstain": examples,
    }

    print("=== LLM-as-Judge Reward-Abstain Summary ===")
    print(f"Total={stats['total']} | Answered={stats['answered']} | Abstained={stats['abstained']} | Coverage={stats['coverage']}")
    print(f"Correct={stats['correct']} | Wrong={stats['wrong']} | Acc(answered)={stats['accuracy_answered']}")
    print(f"Utility = Correct - λ*Wrong + r*Abstain = {stats['correct']} - {lam_wrong}*{stats['wrong']} + {reward_abstain}*{stats['abstained']} = {util:.3f}")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_metrics", required=True)
    ap.add_argument("--llm_threshold", type=float, default=0.75)
    ap.add_argument("--f1_threshold", type=float, default=0.5)
    ap.add_argument("--bleu_threshold", type=float, default=0.3)
    ap.add_argument("--lambda_wrong", type=float, default=2.0, help="Penalty weight for wrong answers")
    ap.add_argument("--reward_abstain", type=float, default=0.5, help="Reward for abstentions/missing-info")
    ap.add_argument("--print_examples", type=int, default=10)
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    run(args.input_metrics, args.llm_threshold, args.f1_threshold, args.bleu_threshold,
        args.lambda_wrong, args.reward_abstain, args.print_examples, args.output)

if __name__ == "__main__":
    main()
