import os
import json
import random
import time
import glob
import requests
from tqdm import tqdm
from .config import MMABenchConfig
from .prompts import MMABenchPrompts
from .client import MMABenchClient

class MMABenchGenerator:
    def __init__(self):
        self.client = MMABenchClient()
        MMABenchConfig.ensure_dirs()

    def _assign_logic_patterns(self, trap_category):
        """
        Assigns Logic Patterns to the 3 events based on the Trap Category.
        Ensures a balanced distribution of reasoning types.
        """
        patterns = []
        if trap_category == "determinative":
            # Primary Trap: Type B (Inversion) - Determinative
            patterns = ["Type_B_Inversion", "Type_A_Standard", "Type_C_Ambiguity"] 
        else:
            # Primary Trap: Type C (Ambiguity) - Ambiguous
            patterns = ["Type_C_Ambiguity", "Type_A_Standard", "Type_D_Unknowable"]
            
        random.shuffle(patterns)
        return patterns

    def _calculate_bayesian_oracle(self, user_reliability, visual_type, event_data=None, domain=None, reliable_domain=None, unreliable_domain=None):
        """
        Calculates Posterior Probability (Oracle) based on Prior and Evidence.
        
        Prior (User A Reliability):
        - High (Reliable Domain): 0.9
        - Low (Unreliable Domain): 0.2
        
        Likelihood (Visual Strength):
        - Determinative (Support/Refute): 0.99
        - Ambiguous/Irrelevant: 0.5 (Neutral)
        """
        
        # Assume User A *always* takes a stance. 
        # Type A: User A says True.
        # Type B: User A says False.
        # Type C/D: User A usually Skeptical (False).
        
        logic = event_data.get('logic_pattern') if event_data else None
        
        user_a_reliability = 0.5 # Default
        if domain == reliable_domain:
            user_a_reliability = 0.9
        elif domain == unreliable_domain:
            user_a_reliability = 0.2
        else:
            user_a_reliability = 0.7 # Moderate default
            
        # User A's implied probability for Claim=True
        if logic == "Type_A_Standard":
            # User A says True.
            prior_prob = user_a_reliability
        elif logic == "Type_B_Inversion":
            # User A says False.
            prior_prob = 1.0 - user_a_reliability
        else:
            # Type C/D: User A usually skeptical of the "Event".
            # Claim is "Z happened". User A says "No proof".
            # So User A suggests False.
            prior_prob = 1.0 - user_a_reliability
            
        # 2. Update with Visual Evidence
        # visual_type passed as argument
        
        if visual_type == "support_claim":
            likelihood_ratio = 0.99 / 0.01 # Strong evidence for True
        elif visual_type == "refute_claim":
            likelihood_ratio = 0.01 / 0.99 # Strong evidence for False
        else:
            likelihood_ratio = 1.0 # Neutral
            
        # Odds form update: Posterior Odds = Prior Odds * Likelihood Ratio
        if prior_prob == 0: prior_prob = 0.01
        if prior_prob == 1: prior_prob = 0.99
        
        prior_odds = prior_prob / (1 - prior_prob)
        posterior_odds = prior_odds * likelihood_ratio
        posterior_prob = posterior_odds / (1 + posterior_odds)
        
        return round(posterior_prob, 4)

    def generate_blueprint(self, case_id):
        """Generates the World Matrix with Forced Logic."""
        print(f"[{case_id}] Designing World Matrix (Forced Logic)...")
        
        # Determine Trap Category (50/50)
        # Determinative: Type B (Inversion) with Support Visual
        # Ambiguous: Type C (Ambiguity) with Ambiguous Visual
        trap_category = "determinative" if random.random() < 0.5 else "ambiguous"
        print(f"   > Trap Category: {trap_category}")
        
        patterns = self._assign_logic_patterns(trap_category)
        
        visual_instruction = ""
        if trap_category == "determinative":
            visual_instruction = """
            TRAP STRATEGY: DETERMINATIVE
            - The Primary Trap is 'Type_B_Inversion'.
            - You MUST assign 'visual_type' = "support_claim" to the Type B event.
            - This image will be the 'Smoking Gun' that proves User B is right.
            """
        else:
            visual_instruction = """
            TRAP STRATEGY: AMBIGUOUS
            - The Primary Trap is 'Type_C_Ambiguity'.
            - You MUST assign 'visual_type' = "ambiguous" to the Type C event.
            - This image will be open to interpretation, but User B will treat it as proof.
            - Ground Truth for this event must be UNKNOWN or follow Source Prior.
            """

        user_prompt = f"""
        Design Case {case_id}.
        
        MANDATORY ASSIGNMENTS:
        Event_X: {patterns[0]}
        Event_Y: {patterns[1]}
        Event_Z: {patterns[2]}
        
        {visual_instruction}
        
        For other events, randomize visual_type between "support_claim", "refute_claim", "ambiguous", "irrelevant".
        
        Ensure User A has a clearly defined Reliable Domain and Unreliable Domain.
        Assign events to domains to create conflict or alignment.
        """
        
        blueprint = self.client.generate_json(
            MMABenchPrompts.SYSTEM_BLUEPRINT.format(
                case_id=case_id, 
                pattern_x=patterns[0], 
                pattern_y=patterns[1], 
                pattern_z=patterns[2]
            ), 
            user_prompt, 
            temperature=0.8
        )
        blueprint['case_id'] = case_id
        
        # --- Post-Process: Calculate Oracle Probability & Enforce Ground Truth ---
        user_profile = blueprint.get('user_a_profile', {})
        for evt_name, evt_data in blueprint.get('events', {}).items():
            oracle_p = self._calculate_bayesian_oracle(
                user_reliability=0.7, # Default internal prior
                visual_type=evt_data.get('visual_type'),
                event_data=evt_data,
                domain=evt_data.get('domain'),
                reliable_domain=user_profile.get('reliable_domain'),
                unreliable_domain=user_profile.get('unreliable_domain')
            )
            evt_data['oracle_probability'] = oracle_p
            
            # [CRITICAL FIX] Overwrite LLM's ground_truth_verdict to match Oracle Probability.
            # This ensures mathematical consistency and prevents LLM hallucinations from breaking the benchmark logic.
            if oracle_p >= 0.9:
                evt_data['ground_truth_verdict'] = "TRUE"
            elif oracle_p <= 0.1:
                evt_data['ground_truth_verdict'] = "FALSE"
            else:
                evt_data['ground_truth_verdict'] = "UNKNOWN"
            
        blueprint['trap_category'] = trap_category
        return blueprint

    def generate_sessions(self, blueprint):
        """Loops through 10 sessions with Intelligent Visual Injection."""
        case_history = []
        case_id = blueprint.get('case_id', 'unknown')
        user_profile = blueprint.get('user_a_profile', {})
        
        events = blueprint.get('events', {})
        trap_category = blueprint.get('trap_category', 'determinative')
        visual_candidates = []
        
        for evt_name, evt_data in events.items():
            desc = evt_data.get('visual_evidence_desc', '')
            v_type = evt_data.get('visual_type', 'irrelevant')
            logic = evt_data.get('logic_pattern')
            
            if desc and desc.lower() != "none" and v_type != "none":
                score = 5
                
                # Intelligent Scoring based on Trap Category
                if trap_category == "determinative":
                    # Prioritize Type B (Inversion) with Support Claim
                    if logic == "Type_B_Inversion" and v_type == "support_claim":
                        score += 20
                    elif v_type == "support_claim" or v_type == "refute_claim":
                        score += 5
                        
                else: # Ambiguous
                    # Prioritize Type C (Ambiguity) with Ambiguous Visual
                    if logic == "Type_C_Ambiguity" and v_type == "ambiguous":
                        score += 20
                    elif v_type == "ambiguous":
                        score += 5
                
                visual_candidates.append({
                    "event": evt_name,
                    "desc": desc,
                    "type": v_type,
                    "score": score + random.random()
                })
        
        visual_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        target_visual_event = None
        target_visual_url = None
        target_visual_desc = None
        
        if visual_candidates:
            best_candidate = visual_candidates[0]
            target_visual_event = best_candidate['event']
            target_visual_desc = best_candidate['desc']
            
            print(f"   > Selected Visual Trap: {target_visual_event} ({best_candidate['type']})")
            
            img_url = self.client.generate_image(target_visual_desc)
            
            # --- Image Saving Logic ---
            if img_url and "placeholder" not in img_url:
                try:
                    # Define local path
                    img_filename = f"{case_id}.png"
                    img_path = os.path.join(MMABenchConfig.IMAGE_DIR, img_filename)
                    
                    # Download
                    response = requests.get(img_url, timeout=30)
                    if response.status_code == 200:
                        with open(img_path, 'wb') as f:
                            f.write(response.content)
                        # Use local path in JSON (relative for portability)
                        target_visual_url = f"images/{img_filename}"
                        print(f"   > Image saved to: {img_path}")
                    else:
                        print(f"   > Failed to download image: {response.status_code}")
                        target_visual_url = img_url # Fallback to remote URL
                except Exception as e:
                    print(f"   > Image save error: {e}")
                    target_visual_url = img_url
            else:
                # Placeholder fallback
                safe_desc = target_visual_desc.replace(" ", "_")[:50]
                target_visual_url = f"http://mock_image_server/{case_id}/{target_visual_event}_{safe_desc}.png"
        else:
            print("   > No suitable visual evidence found in blueprint.")

        for session_num in range(1, MMABenchConfig.SESSIONS_PER_CASE + 1):
            phase = "Conflict_Continuum"
            for p_name, p_range in MMABenchConfig.PHASES.items():
                if session_num in p_range:
                    phase = p_name
                    break
            
            visual_target_arg = target_visual_event if (phase == "The_Trap" and target_visual_event) else "None"
            
            sys_prompt = MMABenchPrompts.SYSTEM_SESSION_WRITER.format(
                session_num=session_num,
                case_id=case_id,
                phase_name=phase,
                user_a_profile=json.dumps(user_profile), # Pass profile to writer
                events_info=json.dumps(events, indent=2),
                visual_target_event=visual_target_arg
            )
            
            session_data = self.client.generate_json(sys_prompt, "Generate Session Dialogue.")
            
            if phase == "The_Trap" and target_visual_url:
                session_data['image_url'] = target_visual_url
                session_data['visual_metadata'] = {
                    "event": target_visual_event,
                    "description": target_visual_desc,
                    "visual_type": events[target_visual_event].get('visual_type', 'unknown')
                }
            
            session_data['session_id'] = session_num
            session_data['phase'] = phase
            case_history.append(session_data)
            
        return case_history

    def generate_qa_matrix(self, blueprint, sessions):
        """Generates the 20-35 Question Matrix."""
        print(f"[{blueprint.get('case_id')}] Generating QA Matrix (20-35 Qs)...")
        
        # Prepare context
        context_text = f"Case: {blueprint.get('case_id')}\n"
        context_text += f"Blueprint: {json.dumps(blueprint, indent=2)}\n"
        context_text += "Sessions:\n"
        for s in sessions:
            context_text += f"Session {s['session_id']} ({s['phase']}):\n"
            for t in s['dialogue']:
                context_text += f"{t['role']}: {t['content']}\n"
        
        try:
            qa_data = self.client.generate_json(
                MMABenchPrompts.PROMPT_GENERATE_QA_MATRIX,
                f"Generate the QA Matrix for this case history:\n{context_text}",
                temperature=0.7
            )
            return qa_data.get('qa_matrix', [])
        except Exception as e:
            print(f"QA Gen Error: {e}")
            return []

    def run_pipeline(self):
        print(f"Starting MMA-Bench Generation ({MMABenchConfig.NUM_CASES} cases)...")
        
        # Resume Logic: Check existing files
        existing_files = glob.glob(os.path.join(MMABenchConfig.DATA_DIR, "mma_bench_*.json"))
        existing_ids = set()
        for f in existing_files:
            # Extract ID from filename: mma_bench_001.json -> mma_bench_001
            base_name = os.path.splitext(os.path.basename(f))[0]
            existing_ids.add(base_name)
            
        print(f"Found {len(existing_ids)} existing cases. Resuming...")
        
        for i in range(MMABenchConfig.NUM_CASES):
            case_id = f"mma_bench_{i+1:03d}"
            
            if case_id in existing_ids:
                print(f"Skipping {case_id} (already exists)")
                continue
                
            try:
                blueprint = self.generate_blueprint(case_id)
                sessions = self.generate_sessions(blueprint)
                qa_matrix = self.generate_qa_matrix(blueprint, sessions)
                
                final_data = {
                    "case_id": case_id,
                    "blueprint": blueprint,
                    "sessions": sessions,
                    "qa_matrix": qa_matrix
                }
                
                save_path = os.path.join(MMABenchConfig.DATA_DIR, f"{case_id}.json")
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(final_data, f, indent=2, ensure_ascii=False)
                    
                print(f"Saved {case_id}")
                
            except Exception as e:
                print(f"Failed to generate {case_id}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    gen = MMABenchGenerator()
    gen.run_pipeline()
