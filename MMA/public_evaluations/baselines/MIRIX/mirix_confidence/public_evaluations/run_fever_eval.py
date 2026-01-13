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


# ------------------------------------------------------------------------------------
# 模块一: 环境与路径引导 (Bootstrap)
# ------------------------------------------------------------------------------------
def _bootstrap_paths_and_env(mirix_path: Optional[str] = None, env_file: Optional[str] = None) -> str:
    """
    - 优先从 --env_file 加载环境变量。
    - 优先将 --mirix_path (用于跑baseline) 注入 sys.path。
    - 否则，自动寻找当前项目的根目录并注入 sys.path。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Warning: python-dotenv not found. Skipping .env file loading.")
        load_dotenv = None

    script_dir = Path(__file__).resolve().parent

    # 1. 环境变量加载（优先 public_evaluations/.env，再回退到仓库根/.env，最后才尝试 baseline/.env）
    env_path = None
    if env_file:
        candidate = Path(env_file).resolve()
        if candidate.exists():
            env_path = candidate
    else:
        default_paths = [
            script_dir / ".env",          # 当前目录/.env（public_evaluations/.env）
            script_dir.parent / ".env",   # 项目根目录/.env
        ]
        if mirix_path:
            mp = Path(mirix_path).resolve()
            default_paths.append(mp / ".env")  # baseline 根目录/.env（兜底）
        for cand in default_paths:
            if cand.exists():
                env_path = cand
                break

    if load_dotenv and env_path:
        load_dotenv(dotenv_path=str(env_path), override=True)
        print(f"Loaded .env from: {env_path}")
    else:
        print("Warning: .env not found; ensure OPENAI_API_KEY is set via environment.")

    # 2. Python 路径注入（优先 baseline mirix）
    path_to_inject = None
    if mirix_path:
        base = Path(mirix_path).resolve()
        if (base / "mirix" / "__init__.py").exists():
            path_to_inject = base
        else:
            print(f"Error: Invalid --mirix_path provided. Could not find 'mirix' package at: {base}")
            sys.exit(1)
    else:
        candidates = [script_dir.parent, script_dir, Path.cwd()]
        for base in candidates:
            if (base / "mirix" / "__init__.py").exists():
                path_to_inject = base
                break

    if path_to_inject:
        if str(path_to_inject) not in sys.path:
            sys.path.insert(0, str(path_to_inject))
        print(f"Resolved and injected mirix package from: {path_to_inject}")
    else:
        print("Error: could not locate 'mirix' package. Run from project root or provide --mirix_path.")
        sys.exit(1)


def _resolve_project_root() -> Path:
    """
    解析项目根目录，以便能定位到 scripts/reset_database.py。
    会尝试多种候选路径，返回第一个命中 reset_database.py 的目录。
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent,           # .../mirix_confidence
        here.parent.parent.parent,    # 项目根目录
        Path.cwd(),                   # 当前工作目录
        Path.cwd().parent,            # 当前工作目录的上一层
    ]
    for cand in candidates:
        if (cand / "scripts" / "reset_database.py").exists():
            return cand
    # 兜底返回当前工作目录
    return Path.cwd()


PROJECT_ROOT = _resolve_project_root()


def _sanitize_and_preflight_openai() -> None:
    """
    清洗 OPENAI_API_KEY 并做一次 OpenAI 预检，提前暴露认证问题。
    若设置了 OPENAI_BASE_URL，则使用该基址；否则使用默认。
    该预检可被 --skip_preflight 关闭。
    """
    try:
        import openai
        from openai import OpenAI
    except Exception:
        print("[WARN] OpenAI SDK 未安装，跳过预检。")
        return

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("[ERROR] 未检测到 OPENAI_API_KEY 环境变量。请在 .env 中设置。")
        return

    cleaned = key.strip().strip('"').strip("'")
    if cleaned != key:
        os.environ["OPENAI_API_KEY"] = cleaned
        print(f"[INFO] 已清洗 OPENAI_API_KEY。len={len(cleaned)}")
    else:
        print(f"[INFO] OPENAI_API_KEY 检测到。len={len(key)}")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        print(f"[INFO] 预检将使用 OPENAI_BASE_URL: {base_url}")

    try:
        # 预检：列出模型（不会消耗配额），某些代理不支持该路由，失败时仅提示不中断评测
        client = OpenAI(base_url=base_url) if base_url else OpenAI()
        _ = client.models.list()
        print("[INFO] OpenAI 预检通过。")
    except Exception as e:
        print(f"[WARN] OpenAI 预检异常：{e}（继续评测）")


# ------------------------------------------------------------------------------------
# 模块二: 数据库与Agent初始化
# ------------------------------------------------------------------------------------
def ensure_sqlite_ready(reset: bool = False, forced_path: Optional[Path] = None) -> Path:
    import sqlite3

    # 允许外部指定 SQLite 路径；否则用默认 ~/.mirix/sqlite.db
    if forced_path is not None:
        db_path = Path(forced_path)
        os.environ["MIRIX_SQLITE_PATH"] = str(db_path)
    else:
        home_dir = Path.home()
        mirix_dir = home_dir / ".mirix"
        db_path = mirix_dir / "sqlite.db"

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
    基线版 AgentWrapper 的 __init__ 签名为 (agent_config_file, load_from=None)。
    因此这里以配置文件路径作为唯一参数创建 AgentWrapper，
    如提供 --model_name，则在初始化后用 set_model 覆盖。
    """
    # 延迟导入，确保 _bootstrap 已执行
    from mirix.agent import AgentWrapper
    print("Initializing Agent...")
    try:
        agent = AgentWrapper(config_path)
        if model_name:
            # 覆盖模型以便基于 CLI 指定模型运行
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


# ------------------------------------------------------------------------------------
# 模块三: FEVER 评测核心逻辑
# ------------------------------------------------------------------------------------
def compose_eval_prompt(claim: str) -> str:
    return (
        "Based solely on the information in your memory, classify the following claim into ONE of three categories: SUPPORTS, REFUTES, or NOT ENOUGH INFO.\n\n"
        f"Claim: \"{claim}\"\n\n"
        "Your answer must be a single word: SUPPORTS, REFUTES, or NOT ENOUGH INFO."
    )


def parse_label_from_text(text: Any) -> str:
    """
    更稳健的标签解析：
    - 取首个非空词，尝试精确匹配三类标签；
    - 若包含完整短语 'NOT ENOUGH INFO' 则判为该类；
    - 否则回退到包含判断。
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
    from mirix.schemas.semantic_memory import SemanticMemoryItem

    header = f"Starting FEVER evaluation: limit={limit}"
    if seed is not None:
        header += f", seed={seed}"
    if formula_mode:
        header += f", formula={formula_mode}"
    print(header)
    dataset = read_fever_jsonl_with_seed(fever_data_path, limit, seed)
    results = []

    # 注意：不要在这里重置数据库（会打断现有连接并造成只读问题）
    # 如需重置，已在 main() 中、Agent 初始化之前完成。

    for i, sample in enumerate(tqdm(dataset, total=len(dataset))):
        claim = sample.get("claim")
        ground_truth_label = sample.get("label")
        evidence_sets = sample.get("evidence", [])

        # 2. “学习”证据
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
                # SemanticMemoryItem 必填字段：name, summary, details, source, tree_path, organization_id
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

        # 3. “提问”声明
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

    # 写最终结果
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(results, out_f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error: failed to write final results: {e}")

    # 原始准确率
    correct_count = sum(1 for r in results if r["is_correct"])
    total_count = len(results)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    # 选择性计分（不改动原始准确率）：CORRECT=1，ABSTAIN(预测为 NEI)=α，其它=0
    abstain_count = sum(1 for r in results if r["predicted_label"] == "NOT ENOUGH INFO")
    selective_sum = sum(1.0 if r["is_correct"] else (abstain_credit if r["predicted_label"] == "NOT ENOUGH INFO" else 0.0) for r in results)
    selective_mean = (selective_sum / total_count) if total_count > 0 else 0.0

    print("\n--- FEVER Evaluation Complete ---")
    print(f"Total samples evaluated: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Abstain (NOT ENOUGH INFO) count: {abstain_count}")
    print(f"SelectiveScore@alpha (alpha={abstain_credit}): {selective_mean:.4f}")


# ------------------------------------------------------------------------------------
# 模块四: 命令行接口 (CLI)
# ------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run FEVER evaluation with Mirix Agent.")
    parser.add_argument("--fever_data_path", type=str, required=True, help="Path to FEVER jsonl file")
    parser.add_argument("--config_path", type=str, required=True, help="Path to Mirix agent config yaml")
    parser.add_argument("--limit", type=int, default=1000, help="Number of samples to evaluate per run")
    parser.add_argument("--output_file", type=str, default="fever_results.json", help="Output JSON file (used when single run)")
    parser.add_argument("--output_dir", type=str, default="fever_runs", help="Directory to store multi-run results")
    parser.add_argument("--reset_sqlite", action="store_true", help="Reset ~/.mirix/sqlite.db before the run")
    parser.add_argument("--env_file", type=str, default=None, help="Explicit path to .env file")
    parser.add_argument("--mirix_path", type=str, default=None, help="Path to the baseline mirix repo to use")
    parser.add_argument("--agent_name", type=str, default="mirix", help="Name of the agent (from config)")
    parser.add_argument("--model_name", type=str, default=None, help="LLM model name to override config")
    parser.add_argument("--skip_preflight", action="store_true", default=True, help="Skip OpenAI preflight models.list()")
    parser.add_argument("--abstain_credit", type=float, default=0.0, help="Partial credit for NOT ENOUGH INFO predictions")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds to run, e.g. 0,1,2")
    parser.add_argument("--formula_modes", type=str, default="st,tc,cs", help="Comma-separated formula modes: st,tc,cs,tri")
    parser.add_argument("--reset_per_run", action="store_true", default=True, help="Reset SQLite before each (formula,seed) run")
    args = parser.parse_args()

    # 1. 动态设置路径和环境变量（修复参数顺序）
    _bootstrap_paths_and_env(mirix_path=args.mirix_path, env_file=args.env_file)

    # 2. 可选：OpenAI 预检（确保认证通过或至少提前暴露问题）
    if not args.skip_preflight:
        _sanitize_and_preflight_openai()
    else:
        print("[INFO] Skip OpenAI preflight per --skip_preflight")

    # 延迟导入，确保路径已设置
    from mirix.schemas.user import User as PydanticUser

    # 解析 seeds 与公式模式
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

    # 输出目录
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 多公式、多种子批跑
    for mode in modes:
        # 设置环境变量并重置置信度模块单例，以确保新模式生效
        os.environ["MIRIX_CONFIDENCE_FORMULA"] = mode
        try:
            from mirix.services.confidence_module import reset_confidence_module
            reset_confidence_module()
        except Exception:
            pass

        for seed in seeds:
            # 每次运行前可选重置 SQLite（确保互不干扰）
            ensure_sqlite_ready(reset=(args.reset_per_run or args.reset_sqlite))

            # 初始化Agent
            agent = initialize_agent(args.agent_name, args.config_path, args.model_name)

            # 构造测试Actor（带种子标记，便于区分）
            random_id = f"user-seed{seed}-{uuid.uuid4().hex[:8]}"
            test_actor = PydanticUser(id=random_id, organization_id="org_fever_test", name="FEVER Tester", timezone="UTC")

            # 输出文件名：公式+种子
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