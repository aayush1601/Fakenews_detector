"""
load_data.py
============
Loads all FakeNewsDB datasets into Oracle XE tables.

Datasets loaded:
  - isot_dataset          (~47 000 rows, ISOT_Fake.csv + ISOT_True.csv)
  - liar_dataset          (~12 800 rows, liar_train.tsv + liar_test.tsv)
  - fnn_pf_dataset        (~2 500 rows,  FakeNewsNet_politifact_*.csv)
  - fnn_gc_dataset        (~22 000 rows, FakeNewsNet_gossipcop_*.csv)

Analytics tables populated from train_results.json (if it exists):
  - experiments
  - model_results
  - bias_scorecard
  - cross_dataset_performance

Usage:
  export ORACLE_USER=fakenews
  export ORACLE_PASSWORD=yourpassword
  export ORACLE_DSN=localhost:1521/XEPDB1

  python load_data.py

Run train_models.py FIRST so that train_results.json is available for
populating model_results and related tables.
"""

import os
import sys
import json
import datetime
import pandas as pd
import oracledb
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"

ISOT_FAKE    = DATA_DIR / "ISOT_Fake.csv"
ISOT_TRUE    = DATA_DIR / "ISOT_True.csv"
LIAR_TRAIN   = DATA_DIR / "liar_train.tsv"
LIAR_TEST    = DATA_DIR / "liar_test.tsv"
FNN_PF_FAKE  = DATA_DIR / "FakeNewsNet_politifact_fake.csv"
FNN_PF_REAL  = DATA_DIR / "FakeNewsNet_politifact_real.csv"
FNN_GC_FAKE  = DATA_DIR / "FakeNewsNet_gossipcop_fake.csv"
FNN_GC_REAL  = DATA_DIR / "FakeNewsNet_gossipcop_real.csv"
TRAIN_RESULTS = BASE_DIR / "train_results.json"

# ── Oracle connection — reads from env vars (same as app.py) ───────────────────
DB_USER = os.getenv("ORACLE_USER",     "fakenews")
DB_PASS = os.getenv("ORACLE_PASSWORD", "yourpassword")
DB_DSN  = os.getenv("ORACLE_DSN",      "localhost:1521/XEPDB1")


def get_conn():
    """Return a new Oracle connection in thin mode (no client libraries needed)."""
    return oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN)


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_str(val, max_len: int) -> str:
    """Convert any value to a truncated string; return '' for NaN/None."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)[:max_len]


def batch_insert(cursor, sql: str, rows: list, batch_size: int = 500) -> int:
    """Execute `sql` in batches of `batch_size`; return total rows inserted."""
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        cursor.executemany(sql, batch)
        total += len(batch)
    return total


def liar_binary(label: str) -> str:
    """Map 6-class LIAR labels to binary REAL / FAKE."""
    real_labels = {"true", "mostly-true", "half-true"}
    return "REAL" if str(label).strip().lower() in real_labels else "FAKE"


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_isot(cursor) -> int:
    """
    Load ISOT_Fake.csv + ISOT_True.csv into isot_dataset.

    Schema: id (IDENTITY), title, text, subject, article_date, label
    Note: id is omitted from INSERT — Oracle IDENTITY column auto-fills it.
    """
    print("\n  ▸ Loading isot_dataset …")

    fake_df = pd.read_csv(ISOT_FAKE)
    fake_df["label"] = "FAKE"

    real_df = pd.read_csv(ISOT_TRUE)
    real_df["label"] = "REAL"

    df = pd.concat([fake_df, real_df], ignore_index=True)

    rows = []
    for _, r in df.iterrows():
        rows.append((
            safe_str(r.get("title",   ""), 500),
            safe_str(r.get("text",    ""), 32767),
            safe_str(r.get("subject", ""), 100),
            safe_str(r.get("date",    ""), 50),
            r["label"],
        ))

    sql = """
        INSERT INTO isot_dataset (title, text, subject, article_date, label)
        VALUES (:1, :2, :3, :4, :5)
    """
    n = batch_insert(cursor, sql, rows)
    print(f"     ✅  isot_dataset: {n:,} rows")
    return n


def load_liar(cursor) -> int:
    """
    Load liar_train.tsv + liar_test.tsv into liar_dataset.

    LIAR TSV columns (no header):
      0: id  1: label  2: statement  3: subject  4: speaker
      5: job  6: state  7: party  8-12: counts  13: context

    Schema: id (IDENTITY), statement_id, label, statement,
            subjects, speaker, party, binary_label
    """
    print("\n  ▸ Loading liar_dataset …")

    cols = [
        "stmt_id", "label", "statement", "subjects",
        "speaker",  "job",   "state",     "party",
        "barely_true_ct", "false_ct", "half_true_ct",
        "mostly_true_ct", "pants_on_fire_ct", "context",
    ]

    frames = []
    for path in (LIAR_TRAIN, LIAR_TEST):
        if path.exists():
            frames.append(pd.read_csv(path, sep="\t", header=None,
                                      names=cols, on_bad_lines="skip"))
    if not frames:
        print("     ⚠️  No LIAR files found — skipping.")
        return 0

    df = pd.concat(frames, ignore_index=True)

    rows = []
    for _, r in df.iterrows():
        rows.append((
            safe_str(r.get("stmt_id",   ""), 50),
            safe_str(r.get("label",     ""), 30),
            safe_str(r.get("statement", ""), 32767),
            safe_str(r.get("subjects",  ""), 300),
            safe_str(r.get("speaker",   ""), 150),
            safe_str(r.get("party",     ""), 80),
            liar_binary(r.get("label",  "")),
        ))

    sql = """
        INSERT INTO liar_dataset
            (statement_id, label, statement, subjects, speaker, party, binary_label)
        VALUES (:1, :2, :3, :4, :5, :6, :7)
    """
    n = batch_insert(cursor, sql, rows)
    print(f"     ✅  liar_dataset: {n:,} rows")
    return n


def _load_fnn_table(cursor, table: str,
                    fake_path: Path, real_path: Path) -> int:
    """
    Generic loader for fnn_pf_dataset / fnn_gc_dataset.

    FNN CSV columns: id, news_url, title, tweet_ids
    Schema:          id (IDENTITY), news_id, news_url, title, label
    """
    frames = []
    for path, lbl in ((fake_path, "FAKE"), (real_path, "REAL")):
        if path.exists():
            tmp = pd.read_csv(path, usecols=["id", "news_url", "title"],
                              on_bad_lines="skip")
            tmp["label"] = lbl
            frames.append(tmp)
        else:
            print(f"     ⚠️  {path.name} not found — skipping {lbl} rows.")

    if not frames:
        return 0

    df = pd.concat(frames, ignore_index=True)

    rows = []
    for _, r in df.iterrows():
        rows.append((
            safe_str(r.get("id",       ""), 100),
            safe_str(r.get("news_url", ""), 1000),
            safe_str(r.get("title",    ""), 600),
            r["label"],
        ))

    sql = f"""
        INSERT INTO {table} (news_id, news_url, title, label)
        VALUES (:1, :2, :3, :4)
    """
    return batch_insert(cursor, sql, rows)


def load_fnn_politifact(cursor) -> int:
    print("\n  ▸ Loading fnn_pf_dataset …")
    n = _load_fnn_table(cursor, "fnn_pf_dataset",
                        FNN_PF_FAKE, FNN_PF_REAL)
    print(f"     ✅  fnn_pf_dataset: {n:,} rows")
    return n


def load_fnn_gossipcop(cursor) -> int:
    print("\n  ▸ Loading fnn_gc_dataset …")
    n = _load_fnn_table(cursor, "fnn_gc_dataset",
                        FNN_GC_FAKE, FNN_GC_REAL)
    print(f"     ✅  fnn_gc_dataset: {n:,} rows")
    return n


# ── Analytics tables: populate from train_results.json ────────────────────────

def load_model_results(cursor) -> int:
    """
    Read train_results.json (written by train_models.py) and insert rows into:
      experiments, model_results, bias_scorecard, cross_dataset_performance.

    Returns total rows inserted into model_results.
    """
    if not TRAIN_RESULTS.exists():
        print("\n  ℹ️  train_results.json not found.")
        print("      Run  python train_models.py  first to populate analytics tables.")
        return 0

    with open(TRAIN_RESULTS) as f:
        results = json.load(f)   # {ModelName: {accuracy, f1, precision, recall, file}}

    print(f"\n  ▸ Populating experiments / model_results / bias_scorecard / "
          f"cross_dataset_performance …")

    now = datetime.datetime.now()

    # ── 1. Insert one experiment row and retrieve its auto-generated id ──────────
    cursor.execute(
        """INSERT INTO experiments (experiment_name, dataset_name, model_name,
                                    run_date, notes)
           VALUES (:1, :2, :3, :4, :5)""",
        ["FakeNewsDB Training Run",
         "ISOT",
         ", ".join(results.keys()),
         now,
         "Trained by train_models.py — scikit-learn proxies"]
    )
    # Fetch the identity value Oracle just generated
    cursor.execute("SELECT MAX(id) FROM experiments")
    exp_id = cursor.fetchone()[0]

    # ── 2. Insert model_results rows ───────────────────────────────────────────
    mr_rows = []
    for model_name, r in results.items():
        mr_rows.append((
            exp_id,
            model_name,
            "ISOT",
            round(float(r.get("accuracy",  0)), 4),
            round(float(r.get("f1",        0)), 4),
            round(float(r.get("precision", 0)), 4),
            round(float(r.get("recall",    0)), 4),
            now,
        ))

    mr_sql = """
        INSERT INTO model_results
            (experiment_id, model_name, dataset_name, accuracy, f1_score,
             precision_score, recall_score, created_at)
        VALUES (:1, :2, :3, :4, :5, :6, :7, :8)
    """
    batch_insert(cursor, mr_sql, mr_rows, batch_size=50)
    print(f"     ✅  model_results: {len(mr_rows)} rows")

    # ── 3. bias_scorecard — per-class accuracy from results ───────────────────
    bs_rows = []
    for model_name, r in results.items():
        fake_acc = round(float(r.get("fake_accuracy", r.get("accuracy", 0))), 4)
        real_acc = round(float(r.get("real_accuracy", r.get("accuracy", 0))), 4)
        disparity = round(abs(real_acc - fake_acc), 4)
        bs_rows.append((
            model_name, "ISOT",
            "FAKE vs REAL",
            fake_acc, real_acc, disparity, now,
        ))

    bs_sql = """
        INSERT INTO bias_scorecard
            (model_name, dataset_name, label_bias,
             fake_accuracy, real_accuracy, disparity_score, evaluated_at)
        VALUES (:1, :2, :3, :4, :5, :6, :7)
    """
    batch_insert(cursor, bs_sql, bs_rows, batch_size=50)
    print(f"     ✅  bias_scorecard: {len(bs_rows)} rows")

    # ── 4. cross_dataset_performance — within-dataset only (train=test) ───────
    cdp_rows = []
    for model_name, r in results.items():
        cdp_rows.append((
            model_name, "ISOT", "ISOT",
            round(float(r.get("accuracy", 0)), 4),
            round(float(r.get("f1",       0)), 4),
            now,
        ))

    cdp_sql = """
        INSERT INTO cross_dataset_performance
            (model_name, train_dataset, test_dataset,
             accuracy, f1_score, evaluated_at)
        VALUES (:1, :2, :3, :4, :5, :6)
    """
    batch_insert(cursor, cdp_sql, cdp_rows, batch_size=50)
    print(f"     ✅  cross_dataset_performance: {len(cdp_rows)} rows")

    return len(mr_rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  FakeNewsDB — Oracle Data Loader")
    print(f"  User : {DB_USER}")
    print(f"  DSN  : {DB_DSN}")
    print("=" * 58)

    try:
        conn = get_conn()
    except Exception as e:
        print(f"\n❌  Cannot connect to Oracle: {e}")
        print("    Check ORACLE_USER / ORACLE_PASSWORD / ORACLE_DSN env vars,")
        print("    and make sure Oracle XE is running.")
        sys.exit(1)

    cursor = conn.cursor()

    try:
        counts = {}
        counts["isot_dataset"]      = load_isot(cursor)
        counts["liar_dataset"]      = load_liar(cursor)
        counts["fnn_pf_dataset"]    = load_fnn_politifact(cursor)
        counts["fnn_gc_dataset"]    = load_fnn_gossipcop(cursor)
        counts["model_results"]     = load_model_results(cursor)

        conn.commit()

        print("\n🎉  All data loaded successfully!")
        print("-" * 40)
        for table, n in counts.items():
            if n > 0:
                print(f"    {table:<30} {n:>7,} rows")
        print()

    except Exception as e:
        conn.rollback()
        print(f"\n❌  Error — rolled back: {e}")
        raise

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
