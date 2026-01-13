import os
import json
import glob
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from .config import MMABenchConfig

class ModalityDynamicsAnalyzer:
    def __init__(self, model_name="gpt-4.1-mini"):
        self.model_name = model_name
        self.base_path = MMABenchConfig.RESULTS_DIR
        self.text_path = os.path.join(self.base_path, model_name, "text")
        self.vision_path = os.path.join(self.base_path, model_name, "vision")
        
    def calculate_entropy(self, confidence):
        """
        Calculates Entropy H(p) for a binary/categorical choice based on confidence.
        """
        try:
            p = float(confidence)
            # Clip to avoid log(0)
            p = max(0.001, min(0.999, p))
            return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        except:
            return 1.0 # Max entropy if undefined

    def norm(self, v):
        """
        Robust normalization of verdicts.
        """
        s = str(v).upper().strip()
        # Clean punctuation
        s = s.replace('.', '').replace(',', '')
        
        if s in ["TRUE", "YES", "CORRECT", "T", "VERIFIED", "REAL"]: 
            return "TRUE"
        if s in ["FALSE", "NO", "INCORRECT", "F", "FAKE", "DEBUNKED"]: 
            return "FALSE"
        return "UNKNOWN"

    def get_signal_vectors(self, logic_type):
        """
        Define Theoretical Signals based on Logic Type.
        S_text: The conclusion implied by the Text stream (or the Trap/Claim).
        S_vis:  The conclusion implied by the Vision stream.
        """
        if logic_type == "Type_B_Inversion":
            # Text (User A) says FALSE. Vision (Trap) says TRUE.
            return "FALSE", "TRUE"
        elif logic_type in ["Type_C_Ambiguity", "Type_D_Unknowable"]:
            # Text (Claim) implies TRUE. Vision (Ambiguous/Irrelevant) implies UNKNOWN.
            return "TRUE", "UNKNOWN"
        return None, None

    def load_data(self):
        # Find common files
        text_files = glob.glob(os.path.join(self.text_path, "result_*.json"))
        data_points = []
        
        print(f"Scanning {len(text_files)} text result files...")
        
        for t_path in text_files:
            filename = os.path.basename(t_path)
            v_path = os.path.join(self.vision_path, filename)
            
            if not os.path.exists(v_path):
                continue
                
            with open(t_path, 'r', encoding='utf-8') as f:
                t_data = json.load(f)
            with open(v_path, 'r', encoding='utf-8') as f:
                v_data = json.load(f)
                
            # Iterate through Bonus Metrics (Events)
            if "bonus_metrics" not in t_data or "bonus_metrics" not in v_data:
                continue
                
            t_bonus = t_data["bonus_metrics"]
            v_bonus = v_data["bonus_metrics"]
            
            for event_id in t_bonus:
                if event_id not in v_bonus:
                    continue
                
                t_event = t_bonus[event_id]
                v_event = v_bonus[event_id]
                
                logic_type = t_event.get("logic_pattern")
                
                # FILTER: Analyze B, C, and D
                if logic_type not in ["Type_B_Inversion", "Type_C_Ambiguity", "Type_D_Unknowable"]:
                    continue
                    
                # 1. Get Signal Vectors
                s_text, s_vis = self.get_signal_vectors(logic_type)
                if not s_text: 
                    continue

                # 2. Extract Predictions
                ym_raw = v_event["agent_response"].get("step1_verdict", "UNKNOWN")
                ym = self.norm(ym_raw)
                
                # 3. Determine Dominance
                category = "Confusion"
                if ym == s_text:
                    category = "Text Dominant"
                elif ym == s_vis:
                    category = "Vision Dominant"
                
                # 4. Calculate Entropy
                conf_t = t_event["agent_response"].get("step1_confidence", 0.5)
                h_t = self.calculate_entropy(conf_t)
                
                conf_v = v_event["agent_response"].get("step1_confidence", 0.5)
                h_v = self.calculate_entropy(conf_v)
                
                # Relative Reasoning Uncertainty (Delta H_rel)
                if (h_t + h_v) == 0:
                    delta_h_rel = 0
                else:
                    delta_h_rel = 2 * (h_t - h_v) / (h_t + h_v)
                
                data_points.append({
                    "logic_type": logic_type,
                    "category": category,
                    "delta_h_rel": delta_h_rel,
                    "ym": ym,
                    "s_text": s_text,
                    "s_vis": s_vis
                })
                
        return data_points

    def analyze(self):
        data = self.load_data()
        if not data:
            print("No relevant data found.")
            return

        print(f"\n=== Modality Dynamics Analysis: Distribution & Confidence ===")
        print(f"Model: {self.model_name}")
        print(f"Total Samples: {len(data)}")
        
        # Segment Analysis by Type
        for logic_type in ["Type_B_Inversion", "Type_C_Ambiguity", "Type_D_Unknowable"]:
            subset = [d for d in data if d["logic_type"] == logic_type]
            n = len(subset)
            if n == 0:
                continue
                
            print(f"\n--- {logic_type} (n={n}) ---")
            print(f"Signals: S_text={subset[0]['s_text']}, S_vis={subset[0]['s_vis']}")
            
            # Calculate Distribution
            counts = {"Text Dominant": 0, "Vision Dominant": 0, "Confusion": 0}
            delta_sums = {"Text Dominant": 0.0, "Vision Dominant": 0.0, "Confusion": 0.0}
            
            for d in subset:
                cat = d["category"]
                counts[cat] += 1
                delta_sums[cat] += d["delta_h_rel"]
                
            # Print Stats
            for cat in ["Text Dominant", "Vision Dominant", "Confusion"]:
                count = counts[cat]
                pct = (count / n) * 100
                avg_delta = delta_sums[cat] / count if count > 0 else 0
                
                print(f"{cat:15s}: {count:3d} ({pct:5.1f}%) | Avg Delta H_rel: {avg_delta:+.4f}")
                
            # Interpretation Helper
            print("  > Avg Delta H_rel > 0 implies Text Entropy > Vision Entropy (Vision 'Easier')")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model name (folder in results)")
    args = parser.parse_args()
    
    analyzer = ModalityDynamicsAnalyzer(model_name=args.model)
    analyzer.analyze()