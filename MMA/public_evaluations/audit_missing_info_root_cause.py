import argparse
import json
import os
import re
import difflib
from typing import Any, Dict, List, Tuple

"""Audit NEI root causes by checking answer presence in retrieved chunks."""

ABSTAIN_PATTERNS = [
    r"There is no information",
    r"No information available",
    r"no available information",
    r"There is no specific information",
    r"There is no explicit information",
    r"NOT ENOUGH INFO",
    r"no relevant information",
    r"insufficient information",
]

MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]

def is_abstain(text: str, abstain_regex: re.Pattern) -> bool:
    if not isinstance(text, str):
        return False
    return bool(abstain_regex.search(text))

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pick_results(dir_path: str) -> str:
    r1 = os.path.join(dir_path, "results.json")
    r2 = os.path.join(dir_path, "evaluation_metrics.json")
    if os.path.isfile(r1):
        return r1
    if os.path.isfile(r2):
        return r2
    return ""

def pick_chunks(dir_path: str) -> str:
    c1 = os.path.join(dir_path, "chunk.json")
    c2 = os.path.join(dir_path, "chunks.json")
    if os.path.isfile(c1):
        return c1
    if os.path.isfile(c2):
        return c2
    return ""

def find_run_units(root: str) -> Tuple[List[Tuple[str, str, str]], List[Dict[str, Any]]]:
    run_units: List[Tuple[str, str, str]] = []
    skipped: List[Dict[str, Any]] = []

    root_results = pick_results(root)
    root_chunks = pick_chunks(root)
    if root_results and root_chunks:
        run_units.append((root, root_results, root_chunks))

    if os.path.isdir(root):
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if not os.path.isdir(p):
                continue
            rp = pick_results(p)
            cp = pick_chunks(p)
            if rp and cp:
                run_units.append((p, rp, cp))
            else:
                skipped.append({
                    "subdir": p,
                    "has_results": bool(rp),
                    "has_chunks": bool(cp),
                    "missing": "results.json/evaluation_metrics.json" if not rp else ("chunk(s).json" if not cp else "")
                })
    return run_units, skipped

def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", s.lower())

def tokenize(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t]

def collect_chunk_texts(chunk_json: Any) -> List[str]:
    texts: List[str] = []
    def walk(x: Any):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            texts.append(x)
    walk(chunk_json)
    return texts

def build_aggregate_text(texts: List[str]) -> str:
    return normalize_text(" ".join(texts))

def simple_date_tokens(answer: str) -> List[str]:
    toks = tokenize(answer)
    out: List[str] = []
    for t in toks:
        if t in MONTHS or re.match(r"^\d{1,4}$", t):
            out.append(t)
    return out

def jaccard_token_similarity(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0

def fuzzy_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def answer_match_in_texts(answer: str, agg_text: str, chunk_texts: List[str], fuzzy_thr: float, jaccard_thr: float) -> Tuple[bool, str]:
    if not isinstance(answer, str):
        return (False, "answer_not_string")
    if not answer.strip():
        return (False, "answer_empty")

    ans_norm = normalize_text(answer)
    # Strategy 1: exact normalized substring
    if ans_norm.strip() and ans_norm.strip() in agg_text:
        return (True, "exact_substring")

    # Strategy 2: short tokens all present
    toks = tokenize(answer)
    short_toks = [t for t in toks if len(t) >= 3]
    if short_toks and all(t in agg_text for t in short_toks):
        return (True, "all_short_tokens")

    # Strategy 3: date tokens present (month/year/number)
    dt = simple_date_tokens(answer)
    if dt and all(t in agg_text for t in dt):
        return (True, "date_tokens")

    # Strategy 4: numeric-only answer
    if re.fullmatch(r"\d{1,4}", answer.strip()):
        if answer.strip() in agg_text:
            return (True, "numeric_substring")

    # Strategy 5: fuzzy/Jaccard against each chunk sentence
    best_fuzzy = 0.0
    best_jacc = 0.0
    for s in chunk_texts:
        fs = fuzzy_similarity(answer, s)
        js = jaccard_token_similarity(answer, s)
        best_fuzzy = max(best_fuzzy, fs)
        best_jacc = max(best_jacc, js)
        if fs >= fuzzy_thr or js >= jaccard_thr:
            return (True, f"fuzzy_or_jaccard(fs={fs:.3f}, js={js:.3f})")

    return (False, f"no_match(fs_max={best_fuzzy:.3f}, js_max={best_jacc:.3f})")

def load_results_entries(results_json: Any) -> List[Dict[str, Any]]:
    if isinstance(results_json, list):
        return results_json
    if isinstance(results_json, dict):
        for k in ["results", "detailed_results", "entries", "data", "evaluations"]:
            v = results_json.get(k)
            if isinstance(v, list):
                return v
    return []

def get_category(entry: Dict[str, Any]) -> str:
    meta = entry.get("metadata", {})
    cat = meta.get("category", None)
    if cat is None:
        cat = entry.get("category", None)
    if cat is None:
        return "unknown"
    return str(cat)

def run(v2_root: str, print_examples: int, output: str, fuzzy_thr: float, jaccard_thr: float):
    abstain_regex = re.compile("|".join(ABSTAIN_PATTERNS), re.IGNORECASE)
    run_units, skipped = find_run_units(v2_root)

    if not run_units:
        print(f"No run units found under: {v2_root}")
        if skipped:
            print(f"Skipped {len(skipped)} subdirs. Examples of missing files:")
            for s in skipped[:5]:
                print(f"- {s['subdir']} | has_results={s['has_results']} has_chunks={s['has_chunks']} missing={s['missing']}")
        summary = {
            "subdirs_scanned": [],
            "skipped": skipped,
            "totals": {"missing_info": 0, "retrieved_hit": 0, "retrieved_miss": 0, "unknown": 0},
            "by_category": {},
            "examples": {"retrieved_hit": [], "retrieved_miss": [], "unknown": []}
        }
        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        return

    summary: Dict[str, Any] = {
        "subdirs_scanned": [u[0] for u in run_units],
        "skipped": skipped,
        "totals": {"missing_info": 0, "retrieved_hit": 0, "retrieved_miss": 0, "unknown": 0},
        "by_category": {},  # cat -> {missing_info, hit, miss, unknown}
        "examples": {"retrieved_hit": [], "retrieved_miss": [], "unknown": []}
    }

    for sd, res_path, chunk_path in run_units:
        try:
            results_json = load_json(res_path)
        except Exception as e:
            summary.setdefault("read_errors", []).append({"path": res_path, "error": str(e)})
            continue

        chunk_texts: List[str] = []
        agg_text = ""
        try:
            chunk_json = load_json(chunk_path)
            chunk_texts = collect_chunk_texts(chunk_json)
            agg_text = build_aggregate_text(chunk_texts)
        except Exception as e:
            summary.setdefault("read_errors", []).append({"path": chunk_path, "error": str(e)})

        entries = load_results_entries(results_json)
        for e in entries:
            rsp = e.get("response") or e.get("model_response") or ""
            if not is_abstain(rsp, abstain_regex):
                continue

            cat = get_category(e)
            summary["totals"]["missing_info"] += 1
            summary["by_category"].setdefault(cat, {"missing_info": 0, "hit": 0, "miss": 0, "unknown": 0})
            summary["by_category"][cat]["missing_info"] += 1

            meta = e.get("metadata", {})
            answer = meta.get("answer") or e.get("answer") or ""
            rec = {
                "question": e.get("question") or meta.get("question") or "",
                "response": rsp,
                "answer": answer,
                "evidence": meta.get("evidence"),
                "category": cat,
                "subdir": sd,
            }

            if agg_text:
                ok, how = answer_match_in_texts(str(answer), agg_text, chunk_texts, fuzzy_thr, jaccard_thr)
                rec["match_strategy"] = how
                if ok:
                    summary["totals"]["retrieved_hit"] += 1
                    summary["by_category"][cat]["hit"] += 1
                    if len(summary["examples"]["retrieved_hit"]) < print_examples:
                        summary["examples"]["retrieved_hit"].append(rec)
                else:
                    if not isinstance(answer, str) or not str(answer).strip():
                        summary["totals"]["unknown"] += 1
                        summary["by_category"][cat]["unknown"] += 1
                        if len(summary["examples"]["unknown"]) < print_examples:
                            summary["examples"]["unknown"].append(rec)
                    else:
                        summary["totals"]["retrieved_miss"] += 1
                        summary["by_category"][cat]["miss"] += 1
                        if len(summary["examples"]["retrieved_miss"]) < print_examples:
                            summary["examples"]["retrieved_miss"].append(rec)
            else:
                rec["match_strategy"] = "chunk_unavailable"
                summary["totals"]["unknown"] += 1
                summary["by_category"][cat]["unknown"] += 1
                if len(summary["examples"]["unknown"]) < print_examples:
                    summary["examples"]["unknown"].append(rec)

    # Percentages
    totals = summary["totals"]
    totals["retrieved_hit_rate"] = round(
        (totals["retrieved_hit"] / totals["missing_info"]) if totals["missing_info"] else 0.0, 4
    )
    totals["retrieved_miss_rate"] = round(
        (totals["retrieved_miss"] / totals["missing_info"]) if totals["missing_info"] else 0.0, 4
    )
    totals["unknown_rate"] = round(
        (totals["unknown"] / totals["missing_info"]) if totals["missing_info"] else 0.0, 4
    )

    # Category-level percentages
    by_cat_rates: Dict[str, Dict[str, float]] = {}
    for cat, c in summary["by_category"].items():
        m = c["missing_info"]
        by_cat_rates[cat] = {
            "hit_rate": round((c["hit"] / m) if m else 0.0, 4),
            "miss_rate": round((c["miss"] / m) if m else 0.0, 4),
            "unknown_rate": round((c["unknown"] / m) if m else 0.0, 4),
        }
    summary["by_category_rates"] = by_cat_rates

    print(f"Scanned {len(run_units)} run units, skipped {len(skipped)}.")
    print(f"V2 missing-info total={totals['missing_info']}, "
          f"retrieved_hit={totals['retrieved_hit']} ({totals['retrieved_hit_rate']*100:.1f}%), "
          f"retrieved_miss={totals['retrieved_miss']} ({totals['retrieved_miss_rate']*100:.1f}%), "
          f"unknown={totals['unknown']} ({totals['unknown_rate']*100:.1f}%).")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_root", required=True, help="Path to v2 results root (contains subfolders like 0..9 or is itself a run unit)")
    ap.add_argument("--print_examples", type=int, default=10)
    ap.add_argument("--output", type=str, default="")
    ap.add_argument("--fuzzy_thr", type=float, default=0.62, help="SequenceMatcher ratio threshold for fuzzy match")
    ap.add_argument("--jaccard_thr", type=float, default=0.6, help="Token Jaccard threshold")
    args = ap.parse_args()
    run(args.v2_root, args.print_examples, args.output, args.fuzzy_thr, args.jaccard_thr)

if __name__ == "__main__":
    main()
