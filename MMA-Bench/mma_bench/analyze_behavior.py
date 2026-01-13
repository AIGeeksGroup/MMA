import os
import json
import glob
import numpy as np
from collections import defaultdict
from tabulate import tabulate
from .config import MMABenchConfig

def load_results(model, mode):
    base_dir = os.path.join(MMABenchConfig.RESULTS_DIR, model, mode)
    files = glob.glob(os.path.join(base_dir, "result_*.json"))
    data = []
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
        except:
            pass
    return data

def analyze_behavior(model):
    modes = ["text", "vision"]
    stats = {m: defaultdict(list) for m in modes}
    
    # Structure for analysis:
    # stats[mode]['confidence'] = [0.95, 0.8, ...]
    # stats[mode]['reserve'] = [5, 0, 10...]
    # stats[mode]['type_c_confidence'] = ...
    # stats[mode]['reflection_change_count'] = 0
    # stats[mode]['reflection_good_change'] = 0
    
    metrics = {m: {
        "count": 0,
        "avg_confidence": [],
        "avg_reserve": [],
        "avg_bet_winner": [], # Amount bet on the chosen verdict
        "logic_breakdown": defaultdict(lambda: {"conf": [], "reserve": [], "acc": [], "count": 0}),
        "reflection": {
            "total_changes": 0,
            "positive_changes": 0, # Wrong -> Right
            "negative_changes": 0, # Right -> Wrong
            "neutral_changes": 0   # Wrong -> Wrong
        }
    } for m in modes}

    for mode in modes:
        results = load_results(model, mode)
        for res in results:
            if "bonus_metrics" not in res: continue
            
            for event_key, event in res["bonus_metrics"].items():
                metrics[mode]["count"] += 1
                
                # 1. Confidence & Wager
                step1 = event.get("agent_response", {}).get("step1_confidence", 0)
                step2 = event.get("agent_response", {}).get("step2_allocation", {})
                
                invest_true = step2.get("invest_true", 0)
                invest_false = step2.get("invest_false", 0)
                reserve = step2.get("retain_reserve", 0)
                
                # Determine "Bet Winner" (the amount placed on the verdict chosen in Step 1)
                # Usually Step 1 verdict matches the higher bet, but let's take max bet
                bet_winner = max(invest_true, invest_false)
                
                metrics[mode]["avg_confidence"].append(step1)
                metrics[mode]["avg_reserve"].append(reserve)
                metrics[mode]["avg_bet_winner"].append(bet_winner)
                
                # Logic Breakdown
                logic = event.get("logic_pattern", "Unknown")
                is_correct = (event.get("scores", {}).get("basic_accuracy", 0) == 1.0)
                
                metrics[mode]["logic_breakdown"][logic]["conf"].append(step1)
                metrics[mode]["logic_breakdown"][logic]["reserve"].append(reserve)
                metrics[mode]["logic_breakdown"][logic]["acc"].append(1 if is_correct else 0)
                metrics[mode]["logic_breakdown"][logic]["count"] += 1
                
                # 2. Reflection (Repentance)
                step3 = event.get("agent_response", {}).get("step3_reflection", {})
                change = step3.get("self_correction_verdict", "Same")
                
                ground_truth = event.get("ground_truth", "UNKNOWN")
                step1_verdict = event.get("agent_response", {}).get("step1_verdict", "UNKNOWN")
                
                # Normalize verdicts
                def normalize(v):
                    v = str(v).upper()
                    if "TRUE" in v: return "TRUE"
                    if "FALSE" in v: return "FALSE"
                    return "UNKNOWN"
                
                gt = normalize(ground_truth)
                v1 = normalize(step1_verdict)
                v3 = normalize(change) if change != "Same" else v1
                
                if change != "Same":
                    metrics[mode]["reflection"]["total_changes"] += 1
                    
                    was_correct = (v1 == gt)
                    is_now_correct = (v3 == gt)
                    
                    if not was_correct and is_now_correct:
                        metrics[mode]["reflection"]["positive_changes"] += 1
                    elif was_correct and not is_now_correct:
                        metrics[mode]["reflection"]["negative_changes"] += 1
                    else:
                        metrics[mode]["reflection"]["neutral_changes"] += 1

    # --- Print Comparison Table ---
    print(f"\nAnalysis for Model: {model}\n")
    
    headers = ["Metric", "Text Mode", "Vision Mode", "Delta (V-T)"]
    rows = []
    
    # General Stats
    t_conf = np.mean(metrics["text"]["avg_confidence"])
    v_conf = np.mean(metrics["vision"]["avg_confidence"])
    rows.append(["Avg Confidence (Step 1)", f"{t_conf:.4f}", f"{v_conf:.4f}", f"{v_conf-t_conf:+.4f}"])
    
    t_res = np.mean(metrics["text"]["avg_reserve"])
    v_res = np.mean(metrics["vision"]["avg_reserve"])
    rows.append(["Avg Reserve (Risk Aversion)", f"{t_res:.2f}", f"{v_res:.2f}", f"{v_res-t_res:+.2f}"])
    
    t_bet = np.mean(metrics["text"]["avg_bet_winner"])
    v_bet = np.mean(metrics["vision"]["avg_bet_winner"])
    rows.append(["Avg Bet Size (Gambling)", f"{t_bet:.2f}", f"{v_bet:.2f}", f"{v_bet-t_bet:+.2f}"])
    
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print("\n")
    
    # Logic Specifics (Focus on C/D for Gambling)
    print("### Gambling on Ambiguity (Type C & D)")
    headers_logic = ["Logic", "Mode", "Acc", "Conf", "Reserve", "Gambling Score (Bet/Conf)"]
    rows_logic = []
    
    for logic in ["Type_C_Ambiguity", "Type_D_Unknowable"]:
        for mode in modes:
            d = metrics[mode]["logic_breakdown"][logic]
            if not d["count"]: continue
            acc = np.mean(d["acc"])
            conf = np.mean(d["conf"])
            res = np.mean(d["reserve"])
            bet = 100 - res
            # Gambling Score: High Bet + Low Accuracy is bad gambling. 
            # Simple metric: Avg Bet size.
            rows_logic.append([logic, mode, f"{acc*100:.1f}%", f"{conf:.2f}", f"{res:.1f}", f"{bet:.1f}"])
            
    print(tabulate(rows_logic, headers=headers_logic, tablefmt="github"))
    print("\n")
    
    # Reflection Stats
    print("### Repentance (Self-Correction)")
    headers_ref = ["Metric", "Text Mode", "Vision Mode"]
    r_t = metrics["text"]["reflection"]
    r_v = metrics["vision"]["reflection"]
    
    rows_ref = [
        ["Total Corrections", r_t["total_changes"], r_v["total_changes"]],
        ["Positive (Saved)", r_t["positive_changes"], r_v["positive_changes"]],
        ["Negative (Ruined)", r_t["negative_changes"], r_v["negative_changes"]],
        ["Neutral (Useless)", r_t["neutral_changes"], r_v["neutral_changes"]]
    ]
    print(tabulate(rows_ref, headers=headers_ref, tablefmt="github"))

if __name__ == "__main__":
    analyze_behavior("qwen3-vl-plus")
