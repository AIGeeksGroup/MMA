import sqlite3, json, argparse, textwrap

def analyze(db_path: str, table: str = "semantic_memory", sample: int = 5):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 存在性探测
    c.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in c.fetchall()]
    meta_col = "metadata_" if "metadata_" in cols else ("metadata" if "metadata" in cols else None)
    if meta_col is None:
        print(f"[ERROR] Table '{table}' has no metadata column in {db_path}")
        return

    # 统计总数
    c.execute(f"SELECT COUNT(*) FROM {table}")
    total = c.fetchone()[0]

    # 扫描行
    empties, nonempties = 0, 0
    sample_nonempty, sample_empty = [], []
    c.execute(f"SELECT id, name, summary, {meta_col} FROM {table}")
    for rid, name, summary, meta_str in c.fetchall():
        try:
            meta = json.loads(meta_str) if meta_str else None
        except Exception:
            meta = None
        is_empty = (meta is None) or (isinstance(meta, dict) and len(meta) == 0)
        if is_empty:
            empties += 1
            if len(sample_empty) < sample:
                sample_empty.append((rid, name, summary))
        else:
            nonempties += 1
            if len(sample_nonempty) < sample:
                sample_nonempty.append((rid, name, summary, meta))

    # 嵌入列统计（如果存在）
    emb_cols = [col for col in cols if "embedding" in col]
    emb_missing = {}
    for col in emb_cols:
        c.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL")
        emb_missing[col] = c.fetchone()[0]

    conn.close()

    print(f"\nDB: {db_path}")
    print(f"Table: {table}")
    print(f"Total items: {total}")
    print(f"Metadata empty: {empties}")
    print(f"Metadata non-empty: {nonempties}")

    if emb_cols:
        print("\nEmbedding columns (missing counts):")
        for col in emb_cols:
            print(f"  - {col}: {emb_missing[col]} missing")

    def show_item(prefix, item):
        rid, name, summary = item[:3]
        print(f"\n[{prefix}] id={rid}")
        print(f"Name: {name}")
        print("Summary:", textwrap.shorten(summary or "", width=160, placeholder="..."))
        if len(item) == 4:
            meta = item[3]
            print("Metadata keys:", list(meta.keys()))

    print("\nSample empty items:")
    for itm in sample_empty:
        show_item("EMPTY", itm)

    print("\nSample non-empty items:")
    for itm in sample_nonempty:
        show_item("NONEMPTY", itm)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to sqlite.db")
    ap.add_argument("--table", default="semantic_memory")
    ap.add_argument("--sample", type=int, default=5)
    args = ap.parse_args()
    analyze(args.db, args.table, args.sample)