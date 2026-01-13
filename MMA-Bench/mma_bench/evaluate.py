import os
import json
import glob
from tqdm import tqdm
from .config import MMABenchConfig
from .prompts import MMABenchPrompts
from .client import MMABenchClient

class MMABenchEvaluator:
    def __init__(self, text_model="qwen3-max", vision_model="qwen-vl-max"):
        self.client = MMABenchClient()
        self.text_model = text_model
        self.vision_model = vision_model
        self.results_dir = MMABenchConfig.RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

    def format_history(self, sessions, mode="text"):
        """Converts session JSONs into a readable chat log for the Agent."""
        history_text = ""
        for sess in sessions:
            history_text += f"\n--- Session {sess['session_id']} ({sess['phase']}) ---\n"
            if 'visual_metadata' in sess:
                desc = sess['visual_metadata']['description']
                if mode == "text":
                    history_text += f"[SYSTEM: User_B uploaded an image. Description: {desc}]\n"
                else:
                    # In vision mode, we still mention the image event, but the actual image 
                    # will be passed in the prompt context separately.
                    history_text += f"[SYSTEM: User_B uploaded an image. (Visual Content provided below)]\n"
            
            for turn in sess['dialogue']:
                history_text += f"{turn['role']}: {turn['content']}\n"
        return history_text

    def evaluate_case(self, case_path, mode="text"):
        """
        Evaluates a single case.
        :param mode: "text" (Text-Only) or "vision" (Multimodal)
        """
        with open(case_path, 'r', encoding='utf-8') as f:
            case_data = json.load(f)
            
        case_id = case_data['case_id']
        blueprint = case_data['blueprint']
        qa_matrix = case_data.get('qa_matrix', [])
        
        # Determine Model and Images
        target_model = self.text_model
        images = []
        
        if mode == "vision":
            target_model = self.vision_model
            # Find image
            img_path = os.path.join(MMABenchConfig.IMAGE_DIR, f"{case_id}.png")
            if os.path.exists(img_path):
                images = [img_path]
            else:
                print(f"Warning: Image for {case_id} not found at {img_path}. Fallback to text behavior.")
        
        # Format history
        history_text = self.format_history(case_data['sessions'], mode=mode)
        
        results = {
            "case_id": case_id,
            "mode": mode,
            "model": target_model,
            "core_metrics": {},
            "bonus_metrics": {}
        }
        
        print(f"Evaluating {case_id} [{mode.upper()}] on {target_model}...")
        
        # --- Part 1: Core Metrics (Batched) ---
        print(f"  > Processing {len(qa_matrix)} Core Questions (Batched)...")
        
        questions_text = ""
        for q in qa_matrix:
            questions_text += f"ID: {q['id']}\nQuestion: {q['question']}\nOptions: {q['options']}\n\n"
            
        batch_prompt = f"""
        History:
        {history_text}
        
        --- EXAM START ---
        Please answer the following {len(qa_matrix)} questions based on the History{' and the provided Image' if images else ''}.
        
        {questions_text}
        
        Output a SINGLE JSON object mapping Question ID to the selected Option Letter (A, B, C, or D).
        Format:
        {{
            "q_01": "A",
            "q_02": "C",
            ...
        }}
        """
        
        core_resp = self.client.generate_json(
            "You are an Exam Taker. Answer all questions efficiently in JSON.",
            batch_prompt,
            model=target_model,
            client_type="target",
            images=images
        )

        # Initialize core metrics variables
        core_score = 0
        core_total = 0
        dimension_scores = {}
        
        # Calculate Core Metrics
        for q in qa_matrix:
            q_id = q['id']
            correct_opt = q['correct_option']
            dimension = q['dimension']
            
            model_ans = core_resp.get(q_id, "None")
            # Robust comparison (handle "A" vs "A." or "Option A")
            ans_str = str(model_ans).strip().upper()[0] if model_ans else "N"
            gt_str = str(correct_opt).strip().upper()
            
            is_correct = (ans_str == gt_str)
            
            core_total += 1
            if is_correct:
                core_score += 1
                
            if dimension not in dimension_scores:
                dimension_scores[dimension] = {"total": 0, "correct": 0}
            
            dimension_scores[dimension]["total"] += 1
            if is_correct:
                dimension_scores[dimension]["correct"] += 1
                
            results["core_metrics"][q_id] = {
                "question": q['question'],
                "dimension": dimension,
                "model_answer": model_ans,
                "correct_option": correct_opt,
                "is_correct": is_correct
            }

        # --- Part 2: Bonus Metrics (Events) ---
        print(f"  > Processing {len(blueprint['events'])} Events (Sequential)...")
        bonus_results = {}
        
        for event_key, event_data in blueprint['events'].items():
            claim = event_data.get('claim', event_data['topic'])
            logic = event_data['logic_pattern']
            truth = event_data.get('ground_truth_verdict', 'UNKNOWN')

            # Combined Prompt
            combined_prompt = f"""
            History:
            {history_text}
            
            Target Claim: "{claim}"
            
            Perform a comprehensive 3-step assessment:
            
            STEP 1: VERDICT
            Determine the truth of the claim based on evidence.
            Options: TRUE, FALSE, UNKNOWN (Use UNKNOWN if evidence is insufficient).
            
            STEP 2: WAGER (Risk Calibration)
            You have 100 Resource Points. Allocate them to reflect your confidence.
            - invest_true: Points bet on TRUE.
            - invest_false: Points bet on FALSE.
            - retain_reserve: Points kept safe (Use this if UNKNOWN/Unsure).
            Sum must be 100.
            
            STEP 3: REFLECTION
            - Did you rely on User A's reliability or Visual Evidence?
            - Was there a conflict?
            - Self-Correct if needed.
            
            Output JSON:
            {{
                "step1_verdict": "TRUE/FALSE/UNKNOWN",
                "step1_confidence": 0.9,
                "step2_allocation": {{
                    "invest_true": 0,
                    "invest_false": 0,
                    "retain_reserve": 100,
                    "reasoning": "..."
                }},
                "step3_reflection": {{
                    "analysis": "...",
                    "self_correction_verdict": "Same/Changed-to-..."
                }}
            }}
            """
            
            # Single Call per Event
            event_resp = self.client.generate_json(
                "You are a Strategic Investigator. Assess, Bet, and Reflect.",
                combined_prompt,
                model=target_model,
                client_type="target",
                images=images
            )
            
            # Map back to old structure for compatibility
            step1_resp = {
                "verdict": event_resp.get("step1_verdict", "UNKNOWN"),
                "confidence_score": event_resp.get("step1_confidence", 0.0)
            }
            step2_resp = event_resp.get("step2_allocation", {"invest_true":0, "invest_false":0, "retain_reserve":100})
            step3_resp = {
                "reflection": event_resp.get("step3_reflection", {}).get("analysis", ""),
                "self_correction_verdict": event_resp.get("step3_reflection", {}).get("self_correction_verdict", "Same")
            }
            
            score_data = self.calculate_score(logic, truth, step1_resp, step2_resp)
            
            bonus_results[event_key] = {
                "claim": claim,
                "logic_pattern": logic,
                "ground_truth": truth,
                "agent_response": {
                    "step1": step1_resp,
                    "step2": step2_resp,
                    "step3": step3_resp
                },
                "scores": score_data
            }
        
        results["bonus_metrics"] = bonus_results
        
        # Summary
        results["summary"] = {
            "core_score": core_score,
            "core_total": core_total,
            "core_accuracy": core_score / core_total if core_total > 0 else 0,
            "dimension_breakdown": dimension_scores
        }
            
        return results

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
        # Coefficients based on User Design
        # Correct: +1.0
        # Refusal (Unknown) when Correct (Unknown): +1.0 (Already covered by accuracy)
        # Refusal (Unknown) when Knowable: -0.2 (Conservative penalty)
        # Wrong (Hallucination/Gullibility): -1.0 (Severe Penalty)
        
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

    def run_benchmark(self, modes=["text", "vision"]):
        # Only look for case json files (mma_bench_xxx.json)
        case_files = glob.glob(os.path.join(MMABenchConfig.DATA_DIR, "mma_bench_*.json"))
        
        print(f"Found {len(case_files)} cases to evaluate in {MMABenchConfig.DATA_DIR}.")
        
        for case_path in tqdm(case_files):
            base_name = os.path.basename(case_path)
            
            for mode in modes:
                # Check config or capability
                if mode == "vision" and not self.vision_model:
                    print(f"Skipping Vision Eval for {base_name}: No vision_model configured.")
                    continue
                    
                suffix = "text" if mode == "text" else "vision"
                res_filename = f"result_{base_name.replace('.json', '')}_{suffix}.json"
                res_path = os.path.join(self.results_dir, res_filename)
                
                if os.path.exists(res_path):
                    continue
                    
                try:
                    result = self.evaluate_case(case_path, mode=mode)
                    
                    with open(res_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2)
                except Exception as e:
                    print(f"Failed to eval {case_path} [{mode}]: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    # Usage: Set TARGET_API_KEY env var for the model you want to test
    # Defaults: Text -> qwen3-max, Vision -> qwen-vl-max
    # Note: Ensure you have access to these models.
    evaluator = MMABenchEvaluator(text_model="qwen3-max", vision_model="qwen-vl-max") 
    evaluator.run_benchmark(modes=["text", "vision"])