import json
import re
from pathlib import Path

ROOT = Path(r"d:\program\MIRIX-public_evaluation\MIRIX-public_evaluation\MIRIX-public_evaluation\mirix_confidence\public_evaluations\results")
FILES = ["v1", "v2", "mirix"]

# 分类模式（可按需扩展/调优）
PATTERNS = {
    "ERROR": re.compile(r'^ERROR$', re.IGNORECASE),
    "MISSING_INFO": re.compile(r'(There is no|no record|no recorded|no information|no evidence|no explicit|no indication|no mention)', re.IGNORECASE),
    "TIME_AGGREGATION": re.compile(r'\b(around|early|mid|late|approximately|roughly)\b', re.IGNORECASE),
    "LONG_EXPLANATION_STYLE": re.compile(r'(^|\. )(However|While|Although|In the memories|Based on|It is known)', re.IGNORECASE),
    "POSSIBLE_FACT_MISMATCH": re.compile(r'\b(seems|appears|likely|suggests)\b', re.IGNORECASE),
}

def classify_response(resp: str):
    if PATTERNS["ERROR"].search(resp or ""):
        return "ERROR"
    if PATTERNS["MISSING_INFO"].search(resp or ""):
        return "MISSING_INFO"
    # 其他类按响应风格/时间措辞粗分类（互斥优先级可调整）
    if PATTERNS["TIME_AGGREGATION"].search(resp or ""):
        return "TIME_AGGREGATION"
    if PATTERNS["LONG_EXPLANATION_STYLE"].search(resp or ""):
        return "LONG_EXPLANATION_STYLE"
    if PATTERNS["POSSIBLE_FACT_MISMATCH"].search(resp or ""):
        return "POSSIBLE_FACT_MISMATCH"
    return "OTHER_UNMATCHED"

# --- CHANGE 1: Modified function to accept a 'path' object ---
def load_results(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容结构：可能是对象或列表
    if isinstance(data, list):
        items = data
    else:
        # 优先匹配常见键，其次兜底找字典里的第一个 list 值
        candidates_keys = ("detailed_results", "results", "evaluations", "items", "data")
        items = None
        for key in candidates_keys:
            val = data.get(key)
            if isinstance(val, list):
                items = val
                break
        if items is None:
            list_values = [v for v in data.values() if isinstance(v, list)]
            items = list_values[0] if list_values else []
    return items

def analyze(dir_name: str):
    # --- CHANGE 2: 'path' is now defined in this scope ---
    path = ROOT / dir_name / "evaluation_metrics.json"
    
    # --- CHANGE 3: Pass the 'path' object to the function ---
    items = load_results(path)
    
    print(f"=== {dir_name} ===")
    print(f"Items loaded: {len(items)} (from {path})") # Now 'path' is defined and accessible
    counts = {"TOTAL_LLM0": 0, "ERROR": 0, "MISSING_INFO": 0, "TIME_AGGREGATION": 0, "LONG_EXPLANATION_STYLE": 0, "POSSIBLE_FACT_MISMATCH": 0, "OTHER_UNMATCHED": 0}
    examples = {k: [] for k in counts.keys() if k != "TOTAL_LLM0"}
    for i, item in enumerate(items):
        llm0 = (str(item.get("llm_score", "")).strip() == "0")
        if not llm0:
            continue
        counts["TOTAL_LLM0"] += 1
        resp = str(item.get("response", "")).strip()
        cat = classify_response(resp)
        counts[cat] += 1
        examples[cat].append({
            "idx": i,
            "question": item.get("question", ""),
            "response": resp[:200]  # 截断展示
        })
    # 输出结果
    print(f"=== {dir_name} ===")
    for k, v in counts.items():
        print(f"{k}: {v}")
    # 如需查看示例，可取消下方注释
    # for k, rows in examples.items():
    #     print(f"\n-- {k} examples ({len(rows)}) --")
    #     for r in rows[:10]:
    #         print(f"[{r['idx']}] {r['question']}\n  {r['response']}\n")

def collect_missing_info_questions(dir_name: str):
    path = ROOT / dir_name / "evaluation_metrics.json"
    items = load_results(path)
    questions = set()
    for item in items:
        if str(item.get("llm_score", "")).strip() != "0":
            continue
        resp = str(item.get("response", "")).strip()
        if classify_response(resp) == "MISSING_INFO":
            q = str(item.get("question", "")).strip()
            questions.add(q)
    return questions

def analyze_missing_info_overlap():
    sets = {d: collect_missing_info_questions(d) for d in FILES}
    print("=== MISSING_INFO overlap ===")
    for d in FILES:
        print(f"{d}: {len(sets[d])}")
    inter_all = set.intersection(*sets.values())
    union_all = set.union(*sets.values())
    print(f"Intersection(all): {len(inter_all)}")
    print(f"Union: {len(union_all)}")
    for a, b in [("v1", "v2"), ("v1", "mirix"), ("v2", "mirix")]:
        print(f"Intersection({a},{b}): {len(sets[a] & sets[b])}")
    print("\nCommon questions (sample):")
    for q in list(inter_all)[:20]:
        print(f"- {q}")

if __name__ == "__main__":
    for d in FILES:
        analyze(d)
    analyze_missing_info_overlap()