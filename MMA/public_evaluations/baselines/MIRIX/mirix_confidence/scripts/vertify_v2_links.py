# 文件: scripts/vertify_v2_links.py (最终健壮版 v3)

import argparse
import json
import os # 确保导入os
import sys # 确保导入sys
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import numpy as np
from types import SimpleNamespace
from sqlalchemy import text as sa_text

# --- 关键：将项目根目录添加到Python路径中 ---
# 这样无论您在哪个目录下运行，都能找到mirix模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- 路径修正结束 ---

# --- 导入所有ORM模型以确保SQLAlchemy能正确初始化 ---
from mirix.orm.semantic_memory import SemanticMemoryItem
from mirix.orm.resource_memory import ResourceMemoryItem
from mirix.orm.episodic_memory import EpisodicEvent
from mirix.orm.knowledge_vault import KnowledgeVaultItem
from mirix.orm.procedural_memory import ProceduralMemoryItem
from mirix.orm.organization import Organization
from mirix.orm.user import User
from mirix.orm.agent import Agent
from mirix.orm.block import Block
from mirix.orm.cloud_file_mapping import CloudFileMapping
from mirix.orm.file import FileMetadata # 正确的类名是 FileMetadata
from mirix.orm.message import Message
from mirix.orm.step import Step
from mirix.orm.tool import Tool

from mirix.services.confidence_module import get_confidence_module, initialize_confidence_module

@contextmanager
def db_context(db_path):
    from mirix.orm.sqlalchemy_base import SqlalchemyBase
    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

MAP = {
    "semantic_memory": {
        "cls": SemanticMemoryItem,
        "emb": ["name_embedding", "summary_embedding", "details_embedding"],
        "txt": ["name", "summary", "details", "metadata_"],
    },
    "resource_memory": {
        "cls": ResourceMemoryItem,
        "emb": ["summary_embedding"],
        "txt": ["title", "summary", "content", "metadata_"],
    },
    "episodic_memory": {
        "cls": EpisodicEvent,
        "emb": ["summary_embedding", "details_embedding"],
        "txt": ["summary", "details", "actor", "metadata_"],
    },
    "knowledge_vault": {
        "cls": KnowledgeVaultItem,
        "emb": ["caption_embedding"],
        "txt": ["caption", "secret_value", "metadata_"],
    },
    "procedural_memory": {
        "cls": ProceduralMemoryItem,
        "emb": ["steps_embedding", "summary_embedding"],
        "txt": ["summary", "steps", "metadata_"],
    },
}
def get_existing_columns(session, cls):
    # 获取该表在当前数据库中的实际列集合
    table = cls.__tablename__
    rows = session.execute(sa_text(f"PRAGMA table_info({table})")).fetchall()
    return {row[1] for row in rows}

def _normalize_item(row_dict, emb_fields, txt_fields, table_name):
    # 将 RowMapping 统一为简单对象，并将 ndarray 嵌入转为 list
    # 还原别名列名：semantic_memory_id -> id
    data = {}
    for k, v in dict(row_dict).items():
        if isinstance(k, str) and k.startswith(f"{table_name}_"):
            k = k[len(f"{table_name}_"):]
        data[k] = v

    def to_list(v):
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return list(v)
        if isinstance(v, str):
            s = v.strip()
            # JSON 列表
            if s.startswith("[") or s.startswith("{"):
                try:
                    import json
                    x = json.loads(s)
                    if isinstance(x, list):
                        return [float(t) for t in x]
                    return x
                except Exception:
                    pass
            # 文本分隔的数值
            try:
                vals = [float(x) for x in s.replace(",", " ").split() if x.strip()]
                return vals if vals else None
            except Exception:
                return None
        return None

    # 规范嵌入字段为 list
    for f in emb_fields:
        if f in data:
            data[f] = to_list(data[f])

    # 规范 metadata_ 为 dict
    if "metadata_" in data and isinstance(data["metadata_"], str):
        s = data["metadata_"].strip()
        if s.startswith("{") or s.startswith("["):
            try:
                import json
                data["metadata_"] = json.loads(s)
            except Exception:
                pass

    return SimpleNamespace(**data)

def _load_item_partial(session, cls, iid, emb_fields, txt_fields):
    cols = get_existing_columns(session, cls)
    needed = set(["id", "organization_id"] + emb_fields + txt_fields)
    present = [c for c in needed if c in cols]
    if not present:
        return None
    q = select(*(getattr(cls, c) for c in present)).where(getattr(cls, "id") == iid)
    row = session.execute(q).mappings().first()
    return _normalize_item(row, emb_fields, txt_fields, cls.__tablename__) if row else None

def _load_candidates(session, cls, exclude_id, org_id, emb_fields, txt_fields, limit=5):
    cols = get_existing_columns(session, cls)
    # 先抽样 id，再按存在列取详细字段
    q = select(getattr(cls, "id")).where(getattr(cls, "id") != exclude_id)
    if org_id and "organization_id" in cols:
        q = q.where(getattr(cls, "organization_id") == org_id)
    ids = session.execute(q.order_by(func.random()).limit(limit)).scalars().all()

    select_cols = set(["id"] + emb_fields + txt_fields)
    present = [c for c in select_cols if c in cols]
    items = []
    for cid in ids:
        rq = select(*(getattr(cls, c) for c in present)).where(getattr(cls, "id") == cid)
        row = session.execute(rq).mappings().first()
        if row:
            items.append(_normalize_item(row, emb_fields, txt_fields, cls.__tablename__))
    return items

def sample_and_check(table: str, n: int, db_path: str):
    initialize_confidence_module()
    cm = get_confidence_module()
    spec = MAP[table]
    cls, emb_fields, txt_fields = spec["cls"], spec["emb"], spec["txt"]

    with db_context(db_path) as session:
        item_ids = session.execute(select(cls.id).order_by(func.random()).limit(n)).scalars().all()
        print(f"[{table}] sampled {len(item_ids)} items")

        for iid in item_ids:
            # 替换 session.get，避免旧库缺列导致的报错
            item = _load_item_partial(session, cls, iid, emb_fields, txt_fields)
            if not item:
                continue

            print("\n" + "="*50)
            print(f"DIAGNOSING ITEM ID: {iid}")

            metadata_content = getattr(item, "metadata_", {})
            summary_embedding = getattr(item, "summary_embedding", None)
            summary_embedding_exists = isinstance(summary_embedding, list) and len(summary_embedding) > 0

            print(f"  - Summary: {getattr(item, 'summary', 'N/A')[:100]}...")
            print(f"  - Metadata Content: {metadata_content if isinstance(metadata_content, dict) else {}}")
            print(f"  - Summary Embedding Exists: {summary_embedding_exists}")
            if summary_embedding_exists:
                print(f"    - Sampled Dims: {str(summary_embedding[:10])}...")

            print("\n  --- Link Generation Debug ---")
            org = getattr(item, "organization_id", None)
            candidates = _load_candidates(session, cls, exclude_id=iid, org_id=org, emb_fields=emb_fields, txt_fields=txt_fields, limit=5)

            this_emb = cm._select_embedding(item, emb_fields)
            this_text = cm._select_text(item, txt_fields)
            print(f"  - Base text for similarity: {(this_text or '').replace('\\n', ' ')[:200]}...")
            print(f"  - Base embedding exists: {this_emb is not None}")

            for r in candidates:
                rid = getattr(r, "id", None)

                emb = cm._select_embedding(r, emb_fields)
                txt = cm._select_text(r, txt_fields)

                emb_sim = cm._cosine_similarity(this_emb or [], emb or [])
                txt_sim = cm._text_similarity(this_text, txt)
                weight = cm.link_emb_w * emb_sim + cm.link_text_w * txt_sim

                print(f"  - Comparing with {rid}: emb_sim={emb_sim:.4f}, txt_sim={txt_sim:.4f}, final_weight={weight:.4f}")
            print("  --- End Debug ---")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_path", required=True, help="Path to the sqlite.db file to analyze")
    ap.add_argument("--table", required=True, choices=list(MAP.keys()))
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    sample_and_check(args.table, args.n, args.db_path)