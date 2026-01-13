import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

def _bootstrap_paths_and_env(env_file: str | None = None) -> None:
    """
    - 先加载 .env（默认 public_evaluations/.env，或显式传入 --env_file）
    - 再把包含 mirix/__init__.py 的顶层目录插入 sys.path，确保能 import mirix
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        # 如果没安装 python-dotenv，可根据需要跳过加载
        load_dotenv = None

    script_dir = Path(__file__).resolve().parent

    # 选择 .env 文件：优先脚本同级，其次仓库根；或使用传入参数
    env_path = None
    if env_file:
        candidate = Path(env_file).resolve()
        if candidate.exists():
            env_path = candidate
    else:
        pe_env = script_dir / ".env"
        root_env = script_dir.parent / ".env"
        if pe_env.exists():
            env_path = pe_env
        elif root_env.exists():
            env_path = root_env

    if load_dotenv and env_path and env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
        print(f"Loaded .env from: {env_path}")
    else:
        print("Warning: .env not found or python-dotenv missing; ensure OPENAI_API_KEY is set.")

    # 注入 sys.path：寻找包含 mirix/__init__.py 的目录
    candidates = [
        script_dir,             # public_evaluations
        script_dir.parent,      # 仓库根（常见）
        Path.cwd(),             # 当前工作目录
        Path.cwd().parent,      # 上一级目录（兼容某些调用方式）
    ]
    inserted = False
    for base in candidates:
        if (base / "mirix" / "__init__.py").exists():
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            inserted = True
            print(f"Resolved mirix package at: {base}")
            break
    if not inserted:
        print("Error: could not locate 'mirix' package. Check your working directory layout.")
        sys.exit(1)

# 在导入 mirix 前先执行环境与路径引导
_bootstrap_paths_and_env()

# ------------------------------------------------------------------------------------
# 1) 在导入 mirix 之前，确保 ~/.mirix/sqlite.db 可写（生成并解锁）
#    - 创建 ~/.mirix 与 ~/.mirix/tmp
#    - 如果 sqlite.db 不存在则创建
#    - 设置权限为 0666
#    - 执行一次写入自测（建表/插入/删除）
# ------------------------------------------------------------------------------------
def ensure_sqlite_ready(reset: bool = False) -> Path:
    home_dir = Path.home()
    mirix_dir = home_dir / ".mirix"
    tmp_dir = mirix_dir / "tmp"
    db_path = mirix_dir / "sqlite.db"

    # 创建目录
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 按需重置数据库
    if reset and db_path.exists():
        try:
            db_path.unlink()
            print(f"Removed existing SQLite DB at {db_path}")
        except Exception as e:
            print(f"Warning: failed to remove existing DB: {e}")

    # 如果 DB 不存在就创建
    if not db_path.exists():
        # 用 sqlite3.connect 创建文件，避免只读空文件
        sqlite3.connect(str(db_path)).close()

    # 设置权限为 0666（容器/不同用户下也能写）
    try:
        os.chmod(db_path, 0o666)
    except Exception as e:
        print(f"Warning: failed to chmod {db_path} to 0666: {e}")

    # 写入自测：能否创建表、插入和删除
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS _write_test (id INTEGER PRIMARY KEY, ts TEXT)")
        cur.execute("INSERT INTO _write_test (ts) VALUES (datetime('now'))")
        conn.commit()
        cur.execute("DROP TABLE _write_test")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error: SQLite write self-test failed: {e}")
        print("Hint: Check mount permissions or filesystem (it may be read-only).")
        sys.exit(1)

    print(f"SQLite DB ready at: {db_path}")
    return db_path

# ------------------------------------------------------------------------------------
# 2) 只有在确保 SQLite 可写后，再导入 mirix 并初始化 Agent
#    - AgentWrapper 构造仅接受配置文件路径，不要传入 model_name（之前 TypeError 的根因）
# ------------------------------------------------------------------------------------
def initialize_agent(config_path: str):
    # 延迟导入，确保上面的 ensure_sqlite_ready 已执行
    try:
        from mirix.agent import AgentWrapper
    except Exception as e:
        print(f"Error importing Mirix AgentWrapper: {e}")
        sys.exit(1)

    print("Initializing Agent...")
    try:
        agent = AgentWrapper(agent_config_file=config_path)
        return agent
    except Exception as e:
        # 这里如果仍然出现 SQLite 相关错误，mirix.server.server 会打印“SQLite schema invalid...”
        print(f"Error initializing AgentWrapper: {e}")
        sys.exit(1)

# ------------------------------------------------------------------------------------
# 3) FEVER 评测：读 paper_dev.jsonl 并给 Agent 任务，输出三分类
#    - 响应规范：仅输出 SUPPORTS / REFUTES / NOT ENOUGH INFO 三者之一
#    - 遇到 ERROR 或空响应，默认记为 NOT ENOUGH INFO，避免过拟合错误信息
# ------------------------------------------------------------------------------------
def compose_eval_prompt(claim: str) -> str:
    return (
        "You are verifying factual claims. "
        "Answer with EXACTLY ONE of the following labels, nothing else:\n"
        "- SUPPORTS\n- REFUTES\n- NOT ENOUGH INFO\n\n"
        f"Claim: {claim}\n"
        "Answer:"
    )

def parse_label_from_text(text: str) -> str:
    if not isinstance(text, str):
        return "NOT ENOUGH INFO"
    t = text.strip().upper()
    if "SUPPORTS" in t:
        return "SUPPORTS"
    if "REFUTES" in t:
        return "REFUTES"
    if "NOT ENOUGH INFO" in t or "NOT ENOUGH" in t:
        return "NOT ENOUGH INFO"
    # 兜底：强制到 NOT ENOUGH INFO，避免误判
    return "NOT ENOUGH INFO"

def read_fever_jsonl(path: str, limit: int) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                items.append(obj)
                if 0 < limit <= len(items):
                    break
            except Exception:
                # 跳过坏行
                continue
    return items

def run_fever_eval(
    agent,
    fever_data_path: str,
    limit: int,
    output_file: str
) -> Dict[str, Any]:
    print(f"Starting FEVER evaluation for the first {limit} samples...")
    dataset = read_fever_jsonl(fever_data_path, limit)
    results = []
    correct = 0

    for sample in tqdm(dataset, total=len(dataset)):
        claim = sample.get("claim", "")
        gold = sample.get("label", "").strip().upper()
        prompt = compose_eval_prompt(claim)

        try:
            resp = agent.send_message(message=prompt)
        except Exception as e:
            # 任何运行时异常（包含 DB、LLM、工具链问题），都按“不足信息”
            resp = "ERROR"

        pred = parse_label_from_text(resp)

        # 计算简单准确率（忽略大小写与多余空格）
        if gold in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO") and pred == gold:
            correct += 1

        results.append({
            "id": sample.get("id"),
            "claim": claim,
            "gold_label": gold,
            "predicted_label": pred,
            "raw_response": resp,
        })

    acc = (correct / len(dataset)) if dataset else 0.0
    summary = {
        "total": len(dataset),
        "correct": correct,
        "accuracy": acc,
    }

    # 保存结果
    payload = {
        "summary": summary,
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved FEVER results to {output_file}")
    print(f"Accuracy: {acc:.4f} ({correct}/{len(dataset)})")
    return payload

# ------------------------------------------------------------------------------------
# 4) CLI
# ------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run FEVER evaluation with Mirix Agent.")
    parser.add_argument("--fever_data_path", type=str, required=True, help="Path to FEVER jsonl, e.g., paper_dev.jsonl")
    parser.add_argument("--config_path", type=str, required=True, help="Path to Mirix agent config yaml")
    parser.add_argument("--limit", type=int, default=200, help="Number of samples to evaluate")
    parser.add_argument("--output_file", type=str, default="fever_results.json", help="Output JSON file")
    parser.add_argument("--reset_sqlite", action="store_true", help="Reset ~/.mirix/sqlite.db before evaluation")
    parser.add_argument("--env_file", type=str, default=None, help="Explicit path to .env (if not default)")
    args = parser.parse_args()

    # 先准备 PATH 与 .env（确保 OPENAI_API_KEY 等）
    _bootstrap_paths_and_env(args.env_file)

    # 再准备 SQLite（生成并解锁），避免只读错误
    ensure_sqlite_ready(reset=args.reset_sqlite)

    # 初始化 Agent（不传 model_name 进构造，避免 TypeError）
    agent = initialize_agent(config_path=args.config_path)

    # 跑 FEVER
    run_fever_eval(
        agent=agent,
        fever_data_path=args.fever_data_path,
        limit=args.limit,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()