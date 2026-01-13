import os
import json
import glob
import argparse
from tqdm import tqdm
from .config import MMABenchConfig
from .client import MMABenchClient

class MMABenchInference:
    def __init__(self, target_model="qwen-vl-plus"):
        self.client = MMABenchClient()
        self.target_model = target_model
        self.output_dir = os.path.join(MMABenchConfig.RESULTS_DIR, "predictions")
        os.makedirs(self.output_dir, exist_ok=True)

    def format_history(self, sessions, mode="text"):
        """Converts session JSONs into a readable chat log."""
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

    def run_case(self, case_path, mode="text"):
        with open(case_path, 'r', encoding='utf-8') as f:
            case_data = json.load(f)
            
        case_id = case_data['case_id']
        qa_matrix = case_data.get('qa_matrix', [])
        
        # Determine Model and Images
        images = []
        if mode == "vision":
            img_filename = f"{case_id}.png"
            img_path = os.path.join(MMABenchConfig.IMAGE_DIR, img_filename)
            # Use absolute path to avoid ambiguity
            abs_img_path = os.path.abspath(img_path)
            
            if os.path.exists(abs_img_path):
                images = [abs_img_path]
            else:
                # CRITICAL: Strict Mode for Vision
                # If mode is vision but image is missing, SKIP this case to avoid data contamination.
                print(f"ERROR: Image for {case_id} not found in VISION mode. Skipping...")
                return
        
        history_text = self.format_history(case_data['sessions'], mode=mode)
        
        prediction = {
            "case_id": case_id,
            "mode": mode,
            "model": self.target_model,
            "core_responses": {},
            "bonus_responses": {}
        }
        
        print(f"Inferencing {case_id} [{mode.upper()}] on {self.target_model}...")
        
        # --- Part 1: Core Metrics (Batched) ---
        if qa_matrix:
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
                model=self.target_model,
                client_type="target",
                images=images
            )
            prediction["core_responses"] = core_resp
        
        # --- Part 2: Bonus Metrics (The 3-Step Probe) ---
        blueprint = case_data['blueprint']
        for event_key, event_data in blueprint['events'].items():
            topic = event_data['topic']
            claim = event_data.get('claim', topic)
            if not claim or len(claim) < 5:
                continue

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
            
            event_resp = self.client.generate_json(
                "You are a Strategic Investigator. Assess, Bet, and Reflect.",
                combined_prompt,
                model=self.target_model,
                client_type="target",
                images=images
            )
            prediction["bonus_responses"][event_key] = event_resp

        return prediction

    def run_all(self, modes=["text"]):
        case_files = glob.glob(os.path.join(MMABenchConfig.DATA_DIR, "mma_bench_*.json"))
        print(f"Found {len(case_files)} cases for Inference.")
        
        for case_path in tqdm(case_files):
            base_name = os.path.basename(case_path)
            for mode in modes:
                # --- Resume Logic (Skip if already exists) ---
                save_dir = os.path.join(self.output_dir, self.target_model, mode)
                os.makedirs(save_dir, exist_ok=True)
                save_name = f"pred_{base_name}" 
                save_path = os.path.join(save_dir, save_name)
                
                if os.path.exists(save_path):
                    # --- Advanced Resume Logic ---
                    # Check if the existing file is complete.
                    try:
                        with open(save_path, 'r', encoding='utf-8') as f:
                            existing_pred = json.load(f)
                        
                        is_complete = True
                        
                        # 1. Check Core Responses
                        if not existing_pred.get("core_responses"):
                            is_complete = False
                        
                        # 2. Check Bonus Responses
                        bonus = existing_pred.get("bonus_responses", {})
                        if not bonus:
                            is_complete = False
                        else:
                            # Check if any event result is empty
                            for k, v in bonus.items():
                                if not v: # Empty dict
                                    is_complete = False
                                    break
                        
                        if is_complete:
                            continue
                        else:
                            print(f"Resuming {base_name} [{mode}]: Found incomplete file. Re-running...")
                            
                    except Exception:
                        print(f"Resuming {base_name} [{mode}]: File corrupted. Re-running...")
                        pass

                try:
                    pred = self.run_case(case_path, mode=mode)
                    
                    # If run_case returned None (e.g. strict vision check skipped), do not save empty file
                    if pred:
                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump(pred, f, indent=2, ensure_ascii=False)
                        
                except Exception as e:
                    print(f"Failed Inference {base_name} [{mode}]: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoRe-Bench Inference")
    parser.add_argument("--model", type=str, default="qwen-vl-plus", help="Target Model Name")
    parser.add_argument("--mode", type=str, choices=["text", "vision", "both"], default="text", help="Inference Mode")
    args = parser.parse_args()

    modes = ["text", "vision"] if args.mode == "both" else [args.mode]
    
    inferencer = MMABenchInference(target_model=args.model)
    inferencer.run_all(modes=modes)
