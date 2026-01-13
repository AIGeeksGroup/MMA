import sqlite3
import json
import sys
from pathlib import Path
from collections import Counter

DEFAULT_DB = r"d:\program\MIRIX-public_evaluation\MIRIX-public_evaluation\MIRIX-public_evaluation\mirix_confidence\public_evaluations\results\v2\sqlite.db"

# 可能的文本字段（按优先级顺序）
TEXT_FIELDS = ["summary", "details", "name", "title", "content", "caption", "secret_value", "steps", "actor", "text"]

def connect(db_path: str) -> sqlite3.Connection:
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def list_tables(conn: sqlite3.Connection):
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    return [r["name"] for r in cur.fetchall()]

def table_info(conn: sqlite3.Connection, table: str):
    cur = conn.execute(f"PRAGMA table_info('{table}');")
    # returns rows with columns: cid, name, type, notnull, dflt_value, pk
    rows = cur.fetchall()
    return [{"name": r["name"], "type": r["type"]} for r in rows]

def count_rows(conn: sqlite3.Connection, table: str) -> int:
    cur = conn.execute(f"SELECT COUNT(*) AS c FROM '{table}';")
    return cur.fetchone()["c"]

def select_text_preview(row: sqlite3.Row):
    # 优先使用已知文本字段
    for f in TEXT_FIELDS:
        if f in row.keys():
            val = row[f]
            if isinstance(val, (str, bytes)) and val:
                if isinstance(val, bytes):
                    try:
                        val = val.decode("utf-8", errors="ignore")
                    except Exception:
                        val = str(val)
                return str(val)[:200]
    # 兜底：从所有字符串列里找最长的
    best = ""
    for k in row.keys():
        v = row[k]
        if isinstance(v, (str, bytes)) and v:
            if isinstance(v, bytes):
                try:
                    v = v.decode("utf-8", errors="ignore")
                except Exception:
                    v = str(v)
            s = str(v)
            if len(s) > len(best):
                best = s
    return best[:200] if best else ""

def parse_json_maybe(x):
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, bytes):
        try:
            x = x.decode("utf-8", errors="ignore")
        except Exception:
            x = str(x)
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        try:
            return json.loads(x)
        except Exception:
            # 可能是非 JSON 文本
            return None
    return None

def flatten_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return str(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        for v in value.values():
            parts.append(flatten_text(v))
        return " ".join([p for p in parts if p])
    if isinstance(value, list):
        return " ".join([flatten_text(v) for v in value])
    return str(value)

def extract_tags_from_metadata(metadata_obj):
    tags = []
    if isinstance(metadata_obj, dict):
        for key in ("tags", "keywords"):
            v = metadata_obj.get(key)
            if isinstance(v, list):
                for t in v:
                    s = str(t).strip()
                    if s:
                        tags.append(s)
            elif isinstance(v, str):
                s = v.strip()
                if s:
                    # 逗号分隔的字符串
                    tags.extend([x.strip() for x in s.split(",") if x.strip()])
    return tags

def calc_non_null_ratio(conn, table: str, column: str) -> float:
    try:
        cur = conn.execute(f"SELECT SUM(CASE WHEN \"{column}\" IS NOT NULL THEN 1 ELSE 0 END) AS nn, COUNT(*) AS c FROM '{table}';")
        row = cur.fetchone()
        c = row["c"] or 1
        return (row["nn"] or 0) / c
    except sqlite3.OperationalError:
        return 0.0

def sample_rows(conn, table: str, limit: int = 3):
    try:
        cur = conn.execute(f"SELECT * FROM '{table}' LIMIT {limit};")
        return cur.fetchall()
    except sqlite3.OperationalError:
        return []

def analyze_links_shape(rows, links_col: str):
    key_shapes = Counter()
    total_links = 0
    for row in rows:
        links_obj = parse_json_maybe(row[links_col])
        if isinstance(links_obj, list):
            total_links += len(links_obj)
            for item in links_obj:
                if isinstance(item, dict):
                    keys = tuple(sorted(item.keys()))
                    key_shapes[keys] += 1
                else:
                    key_shapes[(type(item).__name__,)] += 1
        elif isinstance(links_obj, dict):
            keys = tuple(sorted(links_obj.keys()))
            key_shapes[keys] += 1
            total_links += 1
    return total_links, key_shapes

def detect_memory_tables(table_columns_map):
    # 简单启发式：表名或列集与“记忆”特征相似
    candidates = []
    for t, cols in table_columns_map.items():
        col_names = set(c["name"] for c in cols)
        has_links = "links" in col_names
        has_metadata = "metadata_" in col_names
        has_embedding = any("embedding" in c["name"].lower() for c in cols)
        has_text_fields = any(f in col_names for f in TEXT_FIELDS)
        name_hint = any(h in t.lower() for h in ["semantic_memory", "episodic_memory", "knowledge_vault", "resource_memory", "procedural_memory"])
        if name_hint or (has_links and (has_text_fields or has_embedding or has_metadata)):
            candidates.append(t)
    return candidates

def main(db_path: str):
    print(f"[INFO] Analyzing DB: {db_path}")
    conn = connect(db_path)
    tables = list_tables(conn)
    print(f"[INFO] Tables ({len(tables)}): {tables}")

    # 基础统计
    table_counts = {t: count_rows(conn, t) for t in tables}
    print("\n[SUMMARY] Row counts per table:")
    for t in tables:
        print(f"- {t}: {table_counts[t]}")

    # 收集列信息
    table_cols = {t: table_info(conn, t) for t in tables}
    memory_tables = detect_memory_tables(table_cols)
    print(f"\n[INFO] Detected memory-like tables: {memory_tables or 'None'}")

    # 分析每个候选记忆表
    global_tag_counter = Counter()
    for t in memory_tables:
        cols = table_cols[t]
        col_names = [c["name"] for c in cols]
        print(f"\n=== Table: {t} ===")
        print(f"Columns: {col_names}")
        total = table_counts[t]
        print(f"Total rows: {total}")

        # 计算 ratios
        links_ratio = calc_non_null_ratio(conn, t, "links") if "links" in col_names else 0.0
        metadata_ratio = calc_non_null_ratio(conn, t, "metadata_") if "metadata_" in col_names else 0.0
        embedding_cols = [c for c in col_names if "embedding" in c.lower()]
        embedding_ratios = {c: calc_non_null_ratio(conn, t, c) for c in embedding_cols}

        # 文本字段非空比例（只做参考）
        text_ratios = {f: calc_non_null_ratio(conn, t, f) for f in TEXT_FIELDS if f in col_names}

        print(f"links non-null ratio: {links_ratio:.3f}")
        print(f"metadata_ non-null ratio: {metadata_ratio:.3f}")
        if embedding_ratios:
            print("embedding non-null ratio:")
            for c, r in embedding_ratios.items():
                print(f"  - {c}: {r:.3f}")
        if text_ratios:
            print("text non-null ratio:")
            for f, r in text_ratios.items():
                print(f"  - {f}: {r:.3f}")

        # 抽样行
        rows = sample_rows(conn, t, 10)
        # 分析 links 结构
        if "links" in col_names:
            total_links, key_shapes = analyze_links_shape(rows, "links")
            print(f"Sampled links count (10 rows): {total_links}")
            if key_shapes:
                top_shapes = key_shapes.most_common(5)
                print("Top link item key shapes (sample):")
                for keys, cnt in top_shapes:
                    print(f"  - keys={keys} x{cnt}")

        # 展示 3 条样本
        print("\nSamples:")
        for i, row in enumerate(rows[:3]):
            rid = row["id"] if "id" in row.keys() else f"row#{i}"
            preview = select_text_preview(row)
            # metadata tags
            tags = []
            if "metadata_" in col_names:
                meta_obj = parse_json_maybe(row["metadata_"])
                tags = extract_tags_from_metadata(meta_obj)
                global_tag_counter.update(tags)
            # links count
            lc = 0
            if "links" in col_names:
                lo = parse_json_maybe(row["links"])
                if isinstance(lo, list):
                    lc = len(lo)
                elif isinstance(lo, dict):
                    lc = 1
            print(f"- id={rid}, text='{preview}'")
            print(f"  tags={tags[:6]} links_count={lc}")

    # 全局标签汇总
    if global_tag_counter:
        print("\n[GLOBAL] Top tags/keywords across memory tables:")
        for tag, cnt in global_tag_counter.most_common(20):
            print(f"- {tag}: {cnt}")

    print("\n[DONE] Analysis complete.")

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    main(db_path)