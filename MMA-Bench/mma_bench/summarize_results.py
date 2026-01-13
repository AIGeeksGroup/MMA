import os
import json
import glob
import argparse
import numpy as np
from collections import defaultdict
from tabulate import tabulate
from .config import MMABenchConfig

def summarize(model, mode):
    base_dir = os.path.join(MMABenchConfig.RESULTS_DIR, model, mode)
    files = glob.glob(os.path.join(base_dir, "result_*.json"))
    
    if not files:
        print(f"No result files found in {base_dir}")
        return

    print(f"Found {len(files)} result files for {model} [{mode}]")
    
    # --- Aggregators ---
    core_stats = {
        "total": 0,
        "correct": 0,
        "by_dimension": defaultdict(lambda: {"total": 0, "correct": 0})
    }
    
    bonus_stats = {
        "total_events": 0,
        "basic_accuracy": [],
        "core_score": [],
        "resource_value": [],
        "judge_score": [],
        "by_logic": defaultdict(lambda: {
            "count": 0,
            "accuracy": [],
            "core_score": [],
            "resource": []
        })
    }
    
    # --- Processing ---
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 1. Core Metrics
            if "core_metrics" in data:
                for q_id, q_data in data["core_metrics"].items():
                    dim = q_data.get("dimension", "Unknown")
                    is_correct = q_data.get("is_correct", False)
                    
                    core_stats["total"] += 1
                    if is_correct:
                        core_stats["correct"] += 1
                        
                    core_stats["by_dimension"][dim]["total"] += 1
                    if is_correct:
                        core_stats["by_dimension"][dim]["correct"] += 1
            
            # 2. Bonus Metrics
            if "bonus_metrics" in data:
                for event_id, event_data in data["bonus_metrics"].items():
                    scores = event_data.get("scores", {})
                    logic = event_data.get("logic_pattern", "Unknown")
                    
                    if not scores:
                        continue
                        
                    acc = scores.get("basic_accuracy", 0)
                    core = scores.get("core_robustness_score", 0)
                    res = scores.get("final_resource_value", 0)
                    judge = scores.get("judge_reasoning_score", 0)
                    
                    bonus_stats["total_events"] += 1
                    bonus_stats["basic_accuracy"].append(acc)
                    bonus_stats["core_score"].append(core)
                    bonus_stats["resource_value"].append(res)
                    bonus_stats["judge_score"].append(judge)
                    
                    # Logic Breakdown
                    bonus_stats["by_logic"][logic]["count"] += 1
                    bonus_stats["by_logic"][logic]["accuracy"].append(acc)
                    bonus_stats["by_logic"][logic]["core_score"].append(core)
                    bonus_stats["by_logic"][logic]["resource"].append(res)
                    
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    # --- Reporting ---
    print("\n" + "="*50)
    print(f"  MMA-Bench Summary: {model} ({mode})")
    print("="*50 + "\n")
    
    # 1. Core Report
    print("--- [Part 1] Core Contextual Understanding (25 Qs/Case) ---")
    if core_stats["total"] > 0:
        total_acc = (core_stats["correct"] / core_stats["total"]) * 100
        print(f"Overall Accuracy: {total_acc:.2f}% ({core_stats['correct']}/{core_stats['total']})")
        
        headers = ["Dimension", "Accuracy", "Correct/Total"]
        rows = []
        for dim, stats in sorted(core_stats["by_dimension"].items()):
            acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            rows.append([dim, f"{acc:.2f}%", f"{stats['correct']}/{stats['total']}"])
        
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        print("No Core metrics found.")
        
    print("\n")
    
    # 2. Bonus Report
    print("--- [Part 2] 3-Step Probe (Reasoning & Calibration) ---")
    if bonus_stats["total_events"] > 0:
        avg_acc = np.mean(bonus_stats["basic_accuracy"]) * 100
        avg_core = np.mean(bonus_stats["core_score"])
        avg_res = np.mean(bonus_stats["resource_value"])
        avg_judge = np.mean(bonus_stats["judge_score"])
        
        print(f"Total Events Processed: {bonus_stats['total_events']}")
        print(f"Basic Verdict Accuracy: {avg_acc:.2f}%")
        print(f"CoRe Score (Risk-Adj):  {avg_core:.4f} (Range: -1.0 to 1.0)")
        print(f"Avg Resource Value:     {avg_res:.2f} (Start: 100, Max: ~300)")
        print(f"Avg Judge Score:        {avg_judge:.2f} / 5.0")
        
        print("\n[Breakdown by Logic Pattern]")
        headers = ["Logic Pattern", "Count", "Verdict Acc", "CoRe Score", "Avg Resource"]
        rows = []
        for logic, stats in sorted(bonus_stats["by_logic"].items()):
            l_acc = np.mean(stats["accuracy"]) * 100
            l_core = np.mean(stats["core_score"])
            l_res = np.mean(stats["resource"])
            rows.append([logic, stats["count"], f"{l_acc:.2f}%", f"{l_core:.4f}", f"{l_res:.2f}"])
            
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        print("No Bonus metrics found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, default="text")
    args = parser.parse_args()
    
    summarize(args.model, args.mode)
