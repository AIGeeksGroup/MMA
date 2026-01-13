import json
from tqdm import tqdm
import argparse
import uuid
import os
import sys
import re
import subprocess
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


"""FEVER evaluation CLI using MMA agent with optional batch modes."""
def _bootstrap_paths_and_env(mma_path: Optional[str] = None) -> str:
    """
    Bootstraps Python import paths:
    - Prefer injecting `--mma_path` (baseline repo) into `sys.path`.
    - Otherwise, auto-detect the project root and inject it into `sys.path`.
    """
    script_dir = Path(__file__).resolve().parent

    # Python path injection (prefer baseline `mma` if provided)
    path_to_inject = None
    if mma_path:
        base = Path(mma_path).resolve()
        if (base / "mma" / "__init__.py").exists():
            path_to_inject = base
        else:
            print(f"Error: Invalid --mma_path provided. Could not find 'mma' package at: {base}")
            sys.exit(1)
    else:
        candidates = [script_dir.parent, script_dir, Path.cwd()]
        for base in candidates:
            if (base / "mma" / "__init__.py").exists():
                path_to_inject = base
                break

    if path_to_inject:
        if str(path_to_inject) not in sys.path:
            sys.path.insert(0, str(path_to_inject))
        print(f"Resolved and injected mma package from: {path_to_inject}")
    else:
        print("Error: could not locate 'mma' package. Run from project root or provide --mma_path.")
        sys.exit(1)


def _resolve_project_root() -> Path:
    """
    Resolve the project root to locate `scripts/reset_database.py`.
    Tries multiple candidate directories and returns the first that contains the script.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent,           # .../mma_confidence
        here.parent.parent.parent,    # project root
        Path.cwd(),                   # current working directory
        Path.cwd().parent,            # parent of CWD
    ]
    for cand in candidates:
        if (cand / "scripts" / "reset_database.py").exists():
            return cand
    # Fallback to current working directory when not found
    return Path.cwd()


PROJECT_ROOT = _resolve_project_root()


def _sanitize_and_preflight_openai() -> None:
    """
    Sanitize `OPENAI_API_KEY` and perform a lightweight OpenAI preflight check
    to surface authentication issues early.
    If `OPENAI_BASE_URL` is set it will be used, otherwise defaults are used.
    This preflight can be disabled via `--skip_preflight`.
    """
    try:
        import openai
        from openai import OpenAI
    except Exception:
        print("[WARN] OpenAI SDK not installed, skipping preflight.")
        return

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("[ERROR] OPENAI_API_KEY not found. Please set it in your environment variables.")
        return

    cleaned = key.strip().strip('"').strip("'")
    if cleaned != key:
        os.environ["OPENAI_API_KEY"] = cleaned
        print(f"[INFO] OPENAI_API_KEY sanitized. len={len(cleaned)}")
    else:
        print(f"[INFO] OPENAI_API_KEY detected. len={len(key)}")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        print(f"[INFO] Preflight using OPENAI_BASE_URL: {base_url}")

    try:
        # Preflight: list models (non-billing). Some proxies may not support this route; failures are non-fatal.
        client = OpenAI(base_url=base_url) if base_url else OpenAI()
        _ = client.models.list()
        print("[INFO] OpenAI preflight passed.")
    except Exception as e:
        print(f"[WARN] OpenAI preflight error: {e} (continuing)")


 
def ensure_sqlite_ready(reset: bool = False, forced_path: Optional[Path] = None) -> Path:
    import sqlite3

    # Allow overriding SQLite path; otherwise default to ~/.mma/sqlite.db
    if forced_path is not None:
        db_path = Path(forced_path)
        os.environ["MMA_SQLITE_PATH"] = str(db_path)
    else:
        home_dir = Path.home()
        mma_dir = home_dir / ".mma"
        db_path = mma_dir / "sqlite.db"

    db_dir = db_path.parent
    db_dir.mkdir(parents=True, exist_ok=True)

    if reset and db_path.exists():
        try:
            db_path.unlink()
            print(f"Removed existing SQLite DB at {db_path}")
        except Exception as e:
            print(f"Warning: failed to remove existing DB: {e}")

    if not db_path.exists():
        try:
            sqlite3.connect(str(db_path)).close()
        except Exception as e:
            print(f"Warning: failed to create SQLite DB file: {e}")

    print(f"SQLite DB is ready at: {db_path}")
    return db_path


def initialize_agent(agent_name: str, config_path: str, model_name: Optional[str] = None):
    """
    Initialize the baseline `AgentWrapper` whose constructor signature is
    `(agent_config_file, load_from=None)`. We pass only the config path.
    If `--model_name` is provided, override via `set_model` after init.
    """
    # Delayed import after bootstrap
    from mma.agent import AgentWrapper
    print("Initializing Agent...")
    try:
        agent = AgentWrapper(config_path)
        if model_name:
            # Override model via CLI flag if provided
            agent.set_model(model_name)
            try:
                agent.set_memory_model(model_name)
            except Exception:
                pass
        return agent
    except Exception as e:
        print(f"Error initializing AgentWrapper: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


 
def compose_eval_prompt(claim: str) -> str:
    return (
        "Based solely on the information in your memory, classify the following claim into ONE of three categories: SUPPORTS, REFUTES, or NOT ENOUGH INFO.\n\n"
        f"Claim: \"{claim}\"\n\n"
        "Your answer must be a single word: SUPPORTS, REFUTES, or NOT ENOUGH INFO."
    )


def parse_label_from_text(text: Any) -> str:
    """
    Robust label parsing:
    - Take the first non-empty token and try exact match against the 3 labels.
    - If the full phrase 'NOT ENOUGH INFO' is present, return that class.
    - Otherwise fallback to substring containment checks.
    """
    if not isinstance(text, str):
        return "NOT ENOUGH INFO"
    t = text.strip().upper()
    if "NOT ENOUGH INFO" in t:
        return "NOT ENOUGH INFO"
    tokens = [tok for tok in re.split(r"\W+", t) if tok]
    if tokens:
        first = tokens[0]
        if first in {"SUPPORTS", "REFUTES"}:
            return first
    if "SUPPORT" in t:
        return "SUPPORTS"
    if "REFUTE" in t:
        return "REFUTES"
    return "NOT ENOUGH INFO"


def read_fever_jsonl(path: str, limit: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    if limit and limit > 0:
        items = items[:]
    return items

def read_fever_jsonl_with_seed(path: str, limit: int, seed: Optional[int]) -> List[Dict[str, Any]]:
    items = read_fever_jsonl(path, limit)
    if seed is not None:
        rng = random.Random(int(seed))
        rng.shuffle(items)
    if limit and limit > 0 and len(items) > limit:
        return items[:limit]
    return items


def run_fever_eval(agent, test_actor, fever_data_path: str, limit: int, output_file: str, abstain_credit: float = 0.0, seed: Optional[int] = None, formula_mode: Optional[str] = None):
    from mma.schemas.semantic_memory import SemanticMemoryItem

    header = f"Starting FEVER evaluation: limit={limit}"
    if seed is not None:
        header += f", seed={seed}"
    if formula_mode:
        header += f", formula={formula_mode}"
    print(header)
    dataset = read_fever_jsonl_with_seed(fever_data_path, limit, seed)
    results = []

    # Do not reset database here; handled in `main()` before initialization

    for i, sample in enumerate(tqdm(dataset, total=len(dataset))):
        claim = sample.get("claim")
        ground_truth_label = sample.get("label")
        evidence_sets = sample.get("evidence", [])

    # 2) Ingest evidence
        evidence_sentences: List[str] = []
        if evidence_sets:
            for evidence_set in evidence_sets:
                for ev in evidence_set:
                    if len(ev) > 2 and ev[2]:
                        evidence_sentences.append(ev[2])

        if evidence_sentences:
            items_to_create = []
            sample_id = sample.get("id", i)
            for sent in evidence_sentences:
                # Required fields for `SemanticMemoryItem`: name, summary, details, source, tree_path, organization_id
                items_to_create.append(
                    SemanticMemoryItem(
                        name=f"Evidence for claim {sample_id}",
                        summary=sent,
                        details=sent,
                        source="wikipedia_evidence",
                        tree_path=["fever", "wikipedia", f"claim_{sample_id}"],
                        organization_id=test_actor.organization_id,
                    )
                )
            try:
                agent.client.server.semantic_memory_manager.create_many_items(items=items_to_create, actor=test_actor)
            except Exception as e:
                print(f"Warning: failed to create semantic memory items: {e}")

        # 3) Ask the claim
        prompt = compose_eval_prompt(claim)
        try:
            response = agent.send_message(message=prompt, memorizing=False)
        except Exception as e:
            response = f"ERROR: {e}"
        predicted_label = parse_label_from_text(response)

        results.append({
            "id": sample.get("id"),
            "claim": claim,
            "ground_truth": ground_truth_label,
            "predicted_label": predicted_label,
            "raw_response": response,
            "is_correct": predicted_label == ground_truth_label
        })

        if (i + 1) % 20 == 0:
            try:
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(results, out_f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Warning: failed to write interim results: {e}")

    # Write final results
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(results, out_f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error: failed to write final results: {e}")

    # 1) Basic accuracy
    total_count = len(results)
    
    # 2) Correctness breakdown
    # Correct and actionable (SUPPORTS/REFUTES)
    correct_actionable = sum(1 for r in results if r["is_correct"] and r["predicted_label"] != "NOT ENOUGH INFO")
    # Correct and abstain (NOT ENOUGH INFO) → captures "correct answers that abstained"
    correct_abstain = sum(1 for r in results if r["is_correct"] and r["predicted_label"] == "NOT ENOUGH INFO")
    
    correct_count = correct_actionable + correct_abstain
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    # 3) Abstain statistics
    total_abstain = sum(1 for r in results if r["predicted_label"] == "NOT ENOUGH INFO")

    # 4) Selective Score
    selective_sum = 0.0
    for r in results:
        if r["is_correct"]:
            selective_sum += 1.0
        elif r["predicted_label"] == "NOT ENOUGH INFO":
            selective_sum += abstain_credit
        else:
            selective_sum += 0.0
            
    selective_mean = (selective_sum / total_count) if total_count > 0 else 0.0

    print("\n--- FEVER Evaluation Complete ---")
    print(f"Total samples: {total_count}")
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    print(f"  - Correct Actionable (Supports/Refutes): {correct_actionable}")
    print(f"  - Correct Abstain (True NEI caught): {correct_abstain}")
    print(f"Total Abstain Actions: {total_abstain}")
    print(f"SelectiveScore (alpha={abstain_credit}): {selective_mean:.4f}")


 
def main():
    parser = argparse.ArgumentParser(description="Run FEVER evaluation with MMA Agent.")
    parser.add_argument("--fever_data_path", type=str, required=True, help="Path to FEVER jsonl file")
    parser.add_argument("--config_path", type=str, required=True, help="Path to MMA agent config yaml")
    parser.add_argument("--limit", type=int, default=1000, help="Number of samples to evaluate per run")
    parser.add_argument("--output_file", type=str, default="fever_results.json", help="Output JSON file (used when single run)")
    parser.add_argument("--output_dir", type=str, default="fever_runs", help="Directory to store multi-run results")
    parser.add_argument("--reset_sqlite", action="store_true", help="Reset ~/.mma/sqlite.db before the run")
    parser.add_argument("--mma_path", type=str, default=None, help="Path to the baseline mma repo to use")
    parser.add_argument("--agent_name", type=str, default="mma", help="Name of the agent (from config)")
    parser.add_argument("--model_name", type=str, default=None, help="LLM model name to override config")
    parser.add_argument("--skip_preflight", action="store_true", default=True, help="Skip OpenAI preflight models.list()")
    parser.add_argument("--abstain_credit", type=float, default=0.0, help="Partial credit for NOT ENOUGH INFO predictions")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds to run, e.g. 0,1,2")
    parser.add_argument("--formula_modes", type=str, default="st,tc,cs", help="Comma-separated formula modes: st,tc,cs,tri")
    parser.add_argument("--reset_per_run", action="store_true", default=True, help="Reset SQLite before each (formula,seed) run")
    args = parser.parse_args()

    # 1) Dynamic path and environment bootstrap (fixes import order)
    _bootstrap_paths_and_env(mma_path=args.mma_path)

    # 2) Optional: OpenAI preflight (surface auth issues or confirm setup)
    if not args.skip_preflight:
        _sanitize_and_preflight_openai()
    else:
        print("[INFO] Skip OpenAI preflight per --skip_preflight")

    # Delayed import to ensure sys.path is set
    from mma.schemas.user import User as PydanticUser

    # Parse seeds and formula modes
    def _parse_csv_ints(s: str) -> List[int]:
        vals = []
        for part in (s or "").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                vals.append(int(part))
            except Exception:
                pass
        return vals or [0, 1, 2]

    def _parse_modes(s: str) -> List[str]:
        raw = [x.strip().lower() for x in (s or "").split(",") if x.strip()]
        if not raw:
            raw = ["st", "tc", "cs"]
        valid = {"st", "tc", "cs", "tri"}
        return [m for m in raw if m in valid]

    seeds = _parse_csv_ints(args.seeds)
    modes = _parse_modes(args.formula_modes)

    # Output directory
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Multi-formula, multi-seed batch runs
    for mode in modes:
        # Set environment variable and reset confidence module singleton to apply the new mode
        os.environ["MMA_CONFIDENCE_FORMULA"] = mode
        try:
            from mma.services.confidence_module import reset_confidence_module
            reset_confidence_module()
        except Exception:
            pass

        for seed in seeds:
            # Optionally reset SQLite before each run to avoid cross-run interference
            ensure_sqlite_ready(reset=(args.reset_per_run or args.reset_sqlite))

            # Initialize agent
            agent = initialize_agent(args.agent_name, args.config_path, args.model_name)

            # Construct a test actor (tagged with seed for differentiation)
            random_id = f"user-{uuid.uuid4().hex[:8]}"
            test_actor = PydanticUser(id=random_id, organization_id="org_fever_test", name="FEVER Tester", timezone="UTC")

            # Output file name: formula + seed
            file_name = f"fever_results_{mode}_seed{seed}.json"
            output_path = out_dir / file_name

            run_fever_eval(
                agent=agent,
                test_actor=test_actor,
                fever_data_path=args.fever_data_path,
                limit=args.limit,
                output_file=str(output_path),
                abstain_credit=args.abstain_credit,
                seed=seed,
                formula_mode=mode,
            )


if __name__ == "__main__":
    main()
