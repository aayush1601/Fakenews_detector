"""
app.py
======
Flask backend for FakeNewsDB.

Routes
------
GET  /               -> Serve index.html
POST /query          -> Execute user SQL on Oracle -> return {columns, data, row_count}
POST /predict        -> Classify news text -> return {prediction, confidence, model}
GET  /tables         -> List available Oracle tables with row counts
GET  /schema/<table> -> Return column info for a specific table
GET  /models         -> List available ML models
GET  /health         -> App health / DB connectivity check

Oracle connection (thin mode — no Oracle Client libraries required):
  Uses python-oracledb in thin mode by default.
  Set env vars: ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN

  Example:
    export ORACLE_USER=fakenews
    export ORACLE_PASSWORD=yourpassword
    export ORACLE_DSN=localhost:1521/XEPDB1

  Then:  python app.py
         open http://localhost:5000
"""

import os
import re
import sys

from flask import Flask, request, jsonify, render_template

import oracledb

from model import predict_news, available_models, MODEL_FILES

# ── oracledb thin mode is the default since python-oracledb 1.x.
# Call init_oracle_client() here ONLY if you need thick mode
# (e.g. for advanced features / older Oracle servers):
#   oracledb.init_oracle_client(lib_dir="/path/to/instant-client")

# ── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Oracle config — all read from environment variables ───────────────────────
ORACLE_USER     = os.getenv("ORACLE_USER",     "fakenews")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "yourpassword")
ORACLE_DSN      = os.getenv("ORACLE_DSN",      "localhost:1521/XEPDB1")

# Known FakeNewsDB tables (used for schema listing and safety checks)
KNOWN_TABLES = {
    "isot_dataset",
    "liar_dataset",
    "fnn_pf_dataset",
    "fnn_gc_dataset",
    "experiments",
    "model_results",
    "bias_scorecard",
    "cross_dataset_performance",
    "custom_news",
}

# ── Oracle connection pool ────────────────────────────────────────────────────
_pool = None


def get_pool():
    """Return (or lazily create) the connection pool. Fails gracefully."""
    global _pool
    if _pool is not None:
        return _pool
    try:
        _pool = oracledb.create_pool(
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            dsn=ORACLE_DSN,
            min=2,
            max=10,
            increment=1,
        )
        print("[app.py] Oracle connection pool created.", flush=True)
    except Exception as e:
        print(f"[app.py] WARNING: Could not create Oracle pool: {e}",
              file=sys.stderr)
        _pool = None
    return _pool


def get_connection():
    """
    Acquire a connection from the pool.
    Falls back to a direct connection if the pool is unavailable.
    """
    pool = get_pool()
    if pool:
        return pool.acquire()
    # Direct connect fallback (thin mode, no client libraries needed)
    return oracledb.connect(
        user=ORACLE_USER,
        password=ORACLE_PASSWORD,
        dsn=ORACLE_DSN,
    )


# ── SQL safety filter ─────────────────────────────────────────────────────────
_WRITE_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|"
    r"ATTACH|DETACH|GRANT|REVOKE|COMMIT|ROLLBACK|MERGE|EXECUTE|EXEC)\b",
    re.IGNORECASE,
)


def is_safe_query(sql: str) -> bool:
    """Allow only read-only SELECT statements."""
    stripped = sql.strip().lstrip(";")
    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        return False
    return not bool(_WRITE_RE.search(stripped))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    """Check Oracle connectivity and ML model availability."""
    db_ok  = False
    db_msg = ""
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM DUAL")
            cur.close()
            db_ok  = True
            db_msg = "Oracle connected"
        finally:
            conn.close()
    except Exception as e:
        db_msg = str(e)

    models = available_models()
    return jsonify({
        "status":    "ok" if db_ok else "degraded",
        "database":  {"connected": db_ok, "message": db_msg},
        "ml_models": {"loaded": models, "count": len(models)},
    })


@app.route("/tables")
def list_tables():
    """Return list of known tables with their row counts from Oracle."""
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            tables_info = []
            for tbl in sorted(KNOWN_TABLES):
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {tbl.upper()}")
                    count = cur.fetchone()[0]
                except Exception:
                    count = -1
                tables_info.append({"name": tbl, "rows": count})
            cur.close()
        finally:
            conn.close()
        return jsonify({"tables": tables_info})
    except Exception as e:
        return jsonify({"error": f"Database error: {e}"}), 500


@app.route("/schema/<table_name>")
def table_schema(table_name):
    """Return column names and types for a given table."""
    name_clean = re.sub(r"[^a-zA-Z0-9_]", "", table_name).upper()
    if name_clean.lower() not in KNOWN_TABLES:
        return jsonify({"error": f"Unknown table '{table_name}'"}), 404

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            # Oracle data dictionary query
            cur.execute(
                """SELECT column_name, data_type, data_length, nullable
                   FROM   user_tab_columns
                   WHERE  table_name = :1
                   ORDER  BY column_id""",
                [name_clean],
            )
            cols = [
                {
                    "name":     row[0],
                    "type":     row[1],
                    "length":   row[2],
                    "nullable": row[3],
                }
                for row in cur.fetchall()
            ]
            cur.close()
        finally:
            conn.close()
        return jsonify({"table": name_clean, "columns": cols})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def run_query():
    """
    Execute a read-only SQL SELECT on Oracle.

    Request  JSON: { "sql": "SELECT ...", "limit": 100 }
    Response JSON: { "columns": [...], "data": [[...], ...], "row_count": n,
                     "sql_executed": "..." }
    """
    body  = request.get_json(silent=True) or {}
    sql   = (body.get("sql") or "").strip()
    limit = min(int(body.get("limit", 500)), 2000)

    if not sql:
        return jsonify({"error": "No SQL provided."}), 400

    if not is_safe_query(sql):
        return jsonify({
            "error": (
                "Only read-only SELECT statements are permitted. "
                "Data-modification keywords (INSERT, UPDATE, DELETE, DROP …) "
                "are not allowed."
            )
        }), 400

    # Inject FETCH FIRST if no row limit clause is present (Oracle syntax)
    has_fetch  = bool(re.search(r"\bFETCH\b",  sql, re.IGNORECASE))
    has_rownum = bool(re.search(r"\bROWNUM\b", sql, re.IGNORECASE))
    if not has_fetch and not has_rownum:
        sql = f"{sql.rstrip(';')} FETCH FIRST {limit} ROWS ONLY"

    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql)

            columns = [desc[0].lower() for desc in cur.description]
            rows    = cur.fetchall()

            # Convert Oracle LOB / non-standard objects to plain Python types
            data = []
            for row in rows:
                processed = []
                for cell in row:
                    if hasattr(cell, "read"):       # CLOB / BLOB
                        cell = cell.read()
                    if cell is None:
                        cell = ""
                    if isinstance(cell, (int, float)):
                        processed.append(cell)
                    else:
                        processed.append(str(cell))
                data.append(processed)

            cur.close()
        finally:
            conn.close()

        return jsonify({
            "columns":      columns,
            "data":         data,
            "row_count":    len(data),
            "sql_executed": sql,
        })

    except oracledb.Error as e:
        error_obj, = e.args
        return jsonify({
            "error": f"Oracle error {error_obj.code}: {error_obj.message}"
        }), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Classify a news text as REAL or FAKE.

    Request  JSON: { "text": "...", "model": "LogReg" }
    Response JSON: { "prediction": "REAL|FAKE", "confidence": 92.3,
                     "model": "LogReg" }
    """
    body       = request.get_json(silent=True) or {}
    text       = (body.get("text")  or "").strip()
    model_name = (body.get("model") or "LogReg").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    if model_name not in MODEL_FILES:
        return jsonify({
            "error": (f"Unknown model '{model_name}'. "
                      f"Choose from: {list(MODEL_FILES.keys())}")
        }), 400

    try:
        result = predict_news(text, model_name)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({
            "error": str(e),
            "hint":  "Run 'python train_models.py' to generate model files first.",
        }), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/models")
def list_models():
    """Return available ML models with metadata."""
    descriptions = {
        "NaiveBayes": "Multinomial Naive Bayes with TF-IDF (ngram 1-2). Fast baseline.",
        "LogReg":     "Logistic Regression with TF-IDF (ngram 1-2, sublinear_tf). High accuracy.",
        "BiLSTM":     "SGDClassifier on char n-grams (2-5). Simulates BiLSTM gradient dynamics.",
        "BERT":       "CalibratedLinearSVC on word n-grams (1-3). High-capacity BERT proxy.",
    }
    loaded = available_models()
    models = [
        {
            "name":        name,
            "description": descriptions.get(name, ""),
            "loaded":      name in loaded,
            "file":        f"{MODEL_FILES[name]}.pkl",
        }
        for name in MODEL_FILES
    ]
    return jsonify({"models": models})


@app.route("/api/news", methods=["GET"])
def get_custom_news():
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, title, text, subject, label, TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI:SS') FROM custom_news ORDER BY id DESC")
            data = []
            for r in cur.fetchall():
                text_val = r[2]
                if hasattr(text_val, "read"):
                    text_val = text_val.read()
                data.append({
                    "id": r[0], "title": r[1] or "", "text": text_val or "", 
                    "subject": r[3] or "", "label": r[4], "created_at": r[5]
                })
            cur.close()
        finally:
            conn.close()
        return jsonify({"data": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/news", methods=["POST"])
def create_custom_news():
    body = request.get_json(silent=True) or {}
    title = body.get("title", "")
    text = body.get("text", "")
    subject = body.get("subject", "")
    label = body.get("label", "UNKNOWN")
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO custom_news (title, text, subject, label) VALUES (:1, :2, :3, :4)",
                [title, text, subject, label]
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()
        return jsonify({"success": True}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/news/<int:news_id>", methods=["PUT"])
def update_custom_news(news_id):
    body = request.get_json(silent=True) or {}
    title = body.get("title", "")
    text = body.get("text", "")
    subject = body.get("subject", "")
    label = body.get("label", "UNKNOWN")
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE custom_news SET title = :1, text = :2, subject = :3, label = :4 WHERE id = :5",
                [title, text, subject, label, news_id]
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/news/<int:news_id>", methods=["DELETE"])
def delete_custom_news(news_id):
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM custom_news WHERE id = :1", [news_id])
            conn.commit()
            cur.close()
        finally:
            conn.close()
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 58)
    print("  FakeNewsDB Flask App")
    print(f"  Oracle DSN : {ORACLE_DSN}")
    print(f"  Oracle user: {ORACLE_USER}")
    print("=" * 58)
    app.run(host="0.0.0.0", port=5001, debug=True)
