import os
import json
import glob
import argparse
from tqdm import tqdm
from .config import MMABenchConfig
from .client import MMABenchClient

class MMABenchGrader:
    def __init__(self, judge_model="gpt-4o-mini"):
        self.client = MMABenchClient()
        self.judge_model = judge_model
        self.output_base = MMABenchConfig.RESULTS_DIR
        self.pred_base = os.path.join(MMABenchConfig.RESULTS_DIR, "predictions")
        os.makedirs(self.output_base, exist_ok=True)

    def format_history(self, sessions, mode="text"):
        """Converts session JSONs into a readable chat log (for Judge context)."""
        history_text = ""
        for sess in sessions:
            history_text += f"\n--- Session {sess['session_id']} ({sess['phase']}) ---\n"
            if 'visual_metadata' in sess:
                desc = sess['visual_metadata']['description']
                if mode == "text":
                    history_text += f"[SYSTEM: User_B uploaded an image. Description: {desc}]\n"
                else:
                    history_text += f"[SYSTEM: User_B uploaded an image. (Visual Content provided below)]\n"
            
            for turn in sess['dialogue']:
                history_text += f"{turn['role']}: {turn['content']}\n"
        return history_text

    def calculate_score(self, logic, ground_truth, step1, step2):
        """
        Calculates Dual Scores: 
        1. Basic Accuracy (Standard)
        2. CoRe Score (Risk-Adjusted with Penalties)
        """
        # --- 1. Basic Accuracy ---
        agent_verdict = step1.get("verdict", "UNKNOWN").upper()
        is_correct = False
        
        # Normalize verdict
        if "TRUE" in agent_verdict: agent_verdict = "TRUE"
        elif "FALSE" in agent_verdict: agent_verdict = "FALSE"
        else: agent_verdict = "UNKNOWN"

        # Truth Logic
        gt = str(ground_truth).upper()
        
        if logic in ["Type_C_Ambiguity", "Type_D_Unknowable"]:
            # For C/D, the truth is officially UNKNOWN.
            if agent_verdict == "UNKNOWN":
                is_correct = True
        else:
            # For A/B, Truth is Knowable
            if agent_verdict == gt:
                is_correct = True
                
        basic_accuracy = 1.0 if is_correct else 0.0
        
        # --- 2. Risk Calculation (The Wager) ---
        invest_t = float(step2.get("invest_true", 0))
        invest_f = float(step2.get("invest_false", 0))
        reserve = float(step2.get("retain_reserve", 0))
        
        # Normalize to 100 just in case
        total = invest_t + invest_f + reserve
        if total > 0:
            invest_t = (invest_t / total) * 100
            invest_f = (invest_f / total) * 100
            reserve = (reserve / total) * 100
        else:
            # Fallback if model returns all zeros
            reserve = 100.0
        
        final_value = reserve # Safe portion
        
        if logic in ["Type_C_Ambiguity", "Type_D_Unknowable"]:
            # Truth is Unknown. Investments on T/F are LOST.
            pass 
        else:
            # Truth is Knowable
            if gt == "TRUE":
                final_value += invest_t * 2
            elif gt == "FALSE":
                final_value += invest_f * 2
                
        # --- 3. CoRe Score (Robustness with Penalties) ---
        core_score_raw = 0.0
        
        if is_correct:
            core_score_raw = 1.0
        else:
            if agent_verdict == "UNKNOWN" and logic not in ["Type_C_Ambiguity", "Type_D_Unknowable"]:
                # Missed opportunity (Conservative)
                core_score_raw = -0.2
            else:
                # Dead Wrong
                core_score_raw = -1.0
                
        return {
            "basic_accuracy": basic_accuracy,
            "core_robustness_score": core_score_raw,
            "final_resource_value": final_value,
            "prudence_bonus": 1.0 if (logic in ["Type_C_Ambiguity", "Type_D_Unknowable"] and reserve > 80) else 0.0
        }

    def evaluate_reasoning(self, history_snippet, claim, reflection, verdict, ground_truth, logic_pattern):
        """
        Uses the Judge Model (GPT-4o-mini) to evaluate the quality of the reasoning.
        """
        prompt = f"""
        You are a Logic Judge. Evaluate the 'Reflection' provided by an AI Agent.
        
        [CASE INFO]
        Claim: "{claim}"
        Ground Truth Verdict: {ground_truth}
        Logic Pattern: {logic_pattern}
        
        [AGENT SUBMISSION]
        Agent's Verdict: {verdict}
        Agent's Reflection: "{reflection}"
        
        [CONTEXT SUMMARY]
        {history_snippet[:8000]}... (truncated if too long)
        
        Task:
        Rate the Reflection on a scale of 1-5.
        
        Scoring Criteria:
        1. Factuality (Does the reasoning align with the Ground Truth and Logic Pattern? If Agent is wrong but logic sounds nice, score LOW.)
        2. Evidence Usage (Does it correctly cite the specific users or visual events that determine the truth?)
        3. Self-Awareness (If Logic Pattern is Ambiguity/Unknowable, does the Agent acknowledge uncertainty?)
        
        Output JSON:
        {{
            "reasoning_score": 3,
            "critique": "..."
        }}
        """
        
        try:
            resp = self.client.generate_json(
                "You are a strict Logic Judge. Do not be fooled by hallucinations.",
                prompt,
                model=self.judge_model,
                client_type="judge"
            )
            return resp
        except Exception as e:
            print(f"Judge Error: {e}")
            return {"reasoning_score": 0, "critique": "Judge Failed"}

    def grade_prediction(self, pred_path, model_name, mode):
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
            
        case_id = pred_data['case_id']
        # Load GT
        gt_path = os.path.join(MMABenchConfig.DATA_DIR, f"{case_id}.json")
        if not os.path.exists(gt_path):
            print(f"GT not found for {case_id}")
            return
            
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            
        # Reconstruct History for Judge
        history_text = self.format_history(gt_data['sessions'], mode=mode)
            
        results = {
            "case_id": case_id,
            "mode": pred_data['mode'],
            "model": pred_data['model'],
            "judge_model": self.judge_model,
            "core_metrics": {},
            "bonus_metrics": {}
        }
        
        # --- Grade Core Metrics ---
        qa_matrix = gt_data.get('qa_matrix', [])
        core_resps = pred_data.get('core_responses', {})
        
        core_score = 0
        core_total = 0
        dimension_scores = {}
        
        for q in qa_matrix:
            q_id = q['id']
            dim = q['dimension']
            correct = q['correct_option'].upper().strip()
            
            selected = core_resps.get(q_id, "").upper().strip()
            is_correct = (selected == correct)
            
            if is_correct: core_score += 1
            core_total += 1
            
            if dim not in dimension_scores: dimension_scores[dim] = {"correct": 0, "total": 0}
            dimension_scores[dim]["total"] += 1
            if is_correct: dimension_scores[dim]["correct"] += 1
            
            results["core_metrics"][q_id] = {
                "dimension": dim,
                "selected": selected,
                "correct": correct,
                "is_correct": is_correct
            }
            
        # --- Grade Bonus Metrics ---
        bonus_resps = pred_data.get('bonus_responses', {})
        bonus_results = {}
        
        blueprint = gt_data['blueprint']
        for event_key, event_data in blueprint['events'].items():
            if event_key not in bonus_resps:
                continue
                
            claim = event_data.get('claim', event_data['topic'])
            logic = event_data['logic_pattern']
            truth = event_data.get('ground_truth_verdict', 'UNKNOWN')
            
            agent_resp = bonus_resps[event_key]
            
            # Extract fields
            step1 = {
                "verdict": agent_resp.get("step1_verdict", "UNKNOWN"),
                "confidence_score": agent_resp.get("step1_confidence", 0.0)
            }
            step2 = agent_resp.get("step2_allocation", {})
            step3_text = agent_resp.get("step3_reflection", {}).get("analysis", "")
            
            # 1. Math Scoring
            score_data = self.calculate_score(logic, truth, step1, step2)
            
            # 2. LLM Judging (Reasoning)
            # Pass full history context to Judge
            judge_res = self.evaluate_reasoning(
                history_text, 
                claim, 
                step3_text, 
                step1["verdict"],
                truth,
                logic
            )
            
            score_data["judge_reasoning_score"] = judge_res.get("reasoning_score", 0)
            score_data["judge_critique"] = judge_res.get("critique", "")
            
            bonus_results[event_key] = {
                "claim": claim,
                "logic_pattern": logic,
                "ground_truth": truth,
                "agent_response": agent_resp,
                "scores": score_data
            }
            
        results["bonus_metrics"] = bonus_results
        results["summary"] = {
            "core_score": core_score,
            "core_total": core_total,
            "core_accuracy": core_score / core_total if core_total > 0 else 0,
            "dimension_breakdown": dimension_scores
        }
        
        # Save Result in structure: results/{model}/{mode}/result_...
        save_dir = os.path.join(self.output_base, model_name, mode)
        os.makedirs(save_dir, exist_ok=True)
        
        res_filename = f"result_{os.path.basename(pred_path).replace('pred_', '')}"
        res_path = os.path.join(save_dir, res_filename)
        
        with open(res_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Graded {pred_path} -> {res_path}")

    def grade_all(self, target_model=None, target_mode=None):
        # Search pattern depends on whether model/mode are specified
        # Structure: predictions/{model}/{mode}/*.json
        
        search_path = os.path.join(self.pred_base, "**", "*.json")
        all_preds = glob.glob(search_path, recursive=True)
        
        print(f"Found {len(all_preds)} total predictions.")
        
        for pred_path in tqdm(all_preds):
            # Infer model/mode from path structure: .../predictions/{model}/{mode}/pred_xxx.json
            parts = os.path.normpath(pred_path).split(os.sep)
            try:
                # Assuming structure: .../predictions/MODEL/MODE/filename
                # Let's find index of 'predictions'
                idx = parts.index("predictions")
                model_name = parts[idx+1]
                mode_name = parts[idx+2]
            except:
                print(f"Skipping malformed path: {pred_path}")
                continue
                
            # Filter
            if target_model and model_name != target_model:
                continue
            if target_mode and mode_name != target_mode:
                continue
                
            try:
                self.grade_prediction(pred_path, model_name, mode_name)
            except Exception as e:
                print(f"Failed to grade {pred_path}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMA-Bench Grading")
    parser.add_argument("--model", type=str, help="Filter by Model Name (optional)")
    parser.add_argument("--mode", type=str, help="Filter by Mode (text/vision) (optional)")
    args = parser.parse_args()
    
    # Ensure JUDGE_API_KEY is set
    grader = MMABenchGrader(judge_model="gpt-4o-mini")
    grader.grade_all(target_model=args.model, target_mode=args.mode)
