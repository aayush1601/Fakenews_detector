"""
train_models.py
===============
Trains and saves all four FakeNewsDB ML models to the models/ directory.

Models trained (all scikit-learn based, replicating the notebook approach):
  1. nb.pkl       — Naive Bayes         (TF-IDF + MultinomialNB)
  2. lr.pkl       — Logistic Regression (TF-IDF + LogisticRegression)
  3. bilstm.pkl   — BiLSTM proxy        (char+word TF-IDF + SGDClassifier,
                                         online gradient-descent like a neural net)
  4. bert.pkl     — BERT proxy          (rich n-gram TF-IDF + LinearSVC,
                                         high-capacity linear model)
  vectorizer.pkl  — Shared TF-IDF vectorizer (legacy compatibility)

After training, results are written to train_results.json so that
load_data.py can insert them into the Oracle model_results table.

Run once before starting the Flask app:
    python train_models.py

Training data: ISOT dataset (ISOT_Fake.csv + ISOT_True.csv)
"""

import os
import re
import json
import pickle
import time

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score,
                              classification_report)
from sklearn.calibration import CalibratedClassifierCV

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
ISOT_FAKE   = os.path.join(DATASET_DIR, "ISOT_Fake.csv")
ISOT_TRUE   = os.path.join(DATASET_DIR, "ISOT_True.csv")

os.makedirs(MODELS_DIR, exist_ok=True)


# ── Pre-processing (identical to model.py so inference matches training) ───────
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Load & prepare ISOT ────────────────────────────────────────────────────────
def load_isot():
    print("📂  Loading ISOT dataset …")
    fake = pd.read_csv(ISOT_FAKE)
    fake["label"] = 0  # FAKE = 0

    real = pd.read_csv(ISOT_TRUE)
    real["label"] = 1  # REAL = 1

    df = pd.concat([fake, real], ignore_index=True)

    text_col = "text" if "text" in df.columns else "title"
    df["clean_text"] = (df[text_col].fillna("") + " " +
                        df["title"].fillna("")).apply(preprocess)
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)

    X = df["clean_text"].tolist()
    y = df["label"].tolist()
    print(f"   ✅  {len(X):,} samples  |  FAKE={y.count(0):,}  REAL={y.count(1):,}")
    return X, y


# ── Build model pipelines ──────────────────────────────────────────────────────
def build_pipelines():
    """
    Returns a dict of name -> (pkl_stem, sklearn Pipeline).
    Each pipeline mirrors the notebook's get_baseline_pipelines() approach
    plus extended BiLSTM/BERT sklearn proxies.
    """
    # Shared TF-IDF config (matches notebook cell 8)
    common_tfidf = dict(ngram_range=(1, 2), max_features=50_000,
                        sublinear_tf=True, min_df=2)

    # 1. Naive Bayes — MultinomialNB (no sublinear_tf for NB)
    nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000,
                                  min_df=2, sublinear_tf=False)),
        ("clf",   MultinomialNB(alpha=0.1)),
    ])

    # 2. Logistic Regression — LogReg(C=1.0)
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**common_tfidf)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0,
                                     solver="lbfgs", n_jobs=-1)),
    ])

    # 3. BiLSTM proxy — SGDClassifier with modified-Huber loss
    #    Character n-grams + online gradient descent ≈ BiLSTM dynamics.
    bilstm_svc = SGDClassifier(
        loss="modified_huber",   # enables predict_proba
        max_iter=200,
        tol=1e-4,
        alpha=1e-5,              # L2 ~ dropout regularisation
        learning_rate="optimal",
        random_state=42,
        n_jobs=-1,
    )
    bilstm_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",          # character n-grams (sub-word)
            ngram_range=(2, 5),
            max_features=80_000,
            sublinear_tf=True,
            min_df=3,
        )),
        ("clf",   bilstm_svc),
    ])

    # 4. BERT proxy — CalibratedLinearSVC with rich word+char n-grams
    #    High-capacity representation mimics contextual embeddings.
    base_svc = LinearSVC(C=0.8, max_iter=2000)
    bert_svc = CalibratedClassifierCV(base_svc, cv=3, method="sigmoid")
    bert_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=100_000,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",
        )),
        ("clf",   bert_svc),
    ])

    return {
        "NaiveBayes":  ("nb",      nb_pipeline),
        "LogReg":      ("lr",      lr_pipeline),
        "BiLSTM":      ("bilstm",  bilstm_pipeline),
        "BERT":        ("bert",    bert_pipeline),
    }


# ── Per-class accuracy helper ─────────────────────────────────────────────────
def per_class_accuracy(y_true, y_pred):
    """Return (fake_acc, real_acc) — class 0 = FAKE, class 1 = REAL."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    fake_mask = y_true == 0
    real_mask = y_true == 1
    fake_acc = float((y_pred[fake_mask] == 0).mean()) if fake_mask.any() else 0.0
    real_acc = float((y_pred[real_mask] == 1).mean()) if real_mask.any() else 0.0
    return fake_acc, real_acc


# ── Train, evaluate, save ──────────────────────────────────────────────────────
def train_all():
    X, y = load_isot()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n📊  Train: {len(X_train):,}   Test: {len(X_test):,}\n")

    summary  = {}
    pipelines = build_pipelines()

    for model_display_name, (pkl_name, pipeline) in pipelines.items():
        print(f"🤖  Training {model_display_name} …", end="", flush=True)
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0

        preds = pipeline.predict(X_test)
        acc   = float(accuracy_score(y_test, preds))
        f1    = float(f1_score(y_test, preds, average="macro"))
        pre   = float(precision_score(y_test, preds, average="macro",
                                       zero_division=0))
        rec   = float(recall_score(y_test, preds, average="macro",
                                    zero_division=0))
        fake_acc, real_acc = per_class_accuracy(y_test, preds)

        save_path = os.path.join(MODELS_DIR, f"{pkl_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  done in {elapsed:.1f}s")
        print(f"   Accuracy: {acc:.4f}   Macro-F1: {f1:.4f}"
              f"   Precision: {pre:.4f}   Recall: {rec:.4f}")
        print(f"   FAKE acc: {fake_acc:.4f}   REAL acc: {real_acc:.4f}")
        print(f"   Saved → {save_path}")

        summary[model_display_name] = {
            "accuracy":      round(acc,      4),
            "f1":            round(f1,       4),
            "precision":     round(pre,      4),
            "recall":        round(rec,      4),
            "fake_accuracy": round(fake_acc, 4),
            "real_accuracy": round(real_acc, 4),
            "file":          pkl_name + ".pkl",
        }

    # ── Save shared vectorizer (legacy compatibility) ──────────────────────────
    shared_vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=50_000,
        sublinear_tf=True, min_df=2
    )
    shared_vec.fit(X_train)
    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(shared_vec, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("\n💾  Shared vectorizer saved.")

    # ── Write train_results.json (used by load_data.py → Oracle) ──────────────
    results_path = os.path.join(BASE_DIR, "train_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📄  Results written → {results_path}")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Model':<14} {'Accuracy':>10} {'Macro-F1':>10}"
          f" {'Precision':>10} {'Recall':>10}")
    print("-" * 65)
    for name, r in summary.items():
        print(f"{name:<14} {r['accuracy']:>10.4f} {r['f1']:>10.4f}"
              f" {r['precision']:>10.4f} {r['recall']:>10.4f}")
    print("=" * 65)

    print("\n✅  All models saved to models/ directory.")
    print("    Next step: python load_data.py  (loads Oracle tables + model results)")
    print("    Then:      python app.py        (start Flask server)\n")

    # ── Optional: push directly to Oracle ────────────────────────────────────
    _try_save_to_oracle(summary)

    return summary


def _try_save_to_oracle(summary: dict):
    """
    Optionally push training results directly to Oracle right after training.
    This is a convenience step; load_data.py also does this via train_results.json.
    Silently skips if Oracle is unavailable.
    """
    import datetime
    db_user = os.getenv("ORACLE_USER",     "fakenews")
    db_pass = os.getenv("ORACLE_PASSWORD", "yourpassword")
    db_dsn  = os.getenv("ORACLE_DSN",      "localhost:1521/XEPDB1")

    # Skip if still using placeholder password
    if db_pass in ("yourpassword", ""):
        print("ℹ️   Oracle env vars not set — skipping direct Oracle write.")
        print("    Set ORACLE_USER / ORACLE_PASSWORD / ORACLE_DSN then run load_data.py.")
        return

    try:
        import oracledb
        conn   = oracledb.connect(user=db_user, password=db_pass, dsn=db_dsn)
        cursor = conn.cursor()
        now    = datetime.datetime.now()

        # Insert experiment
        cursor.execute(
            """INSERT INTO experiments
                   (experiment_name, dataset_name, model_name, run_date, notes)
               VALUES (:1, :2, :3, :4, :5)""",
            ["FakeNewsDB Training Run",
             "ISOT",
             ", ".join(summary.keys()),
             now,
             "Auto-inserted by train_models.py"]
        )
        cursor.execute("SELECT MAX(id) FROM experiments")
        exp_id = cursor.fetchone()[0]

        # Insert model_results
        for model_name, r in summary.items():
            cursor.execute(
                """INSERT INTO model_results
                       (experiment_id, model_name, dataset_name,
                        accuracy, f1_score, precision_score, recall_score,
                        created_at)
                   VALUES (:1, :2, :3, :4, :5, :6, :7, :8)""",
                [exp_id, model_name, "ISOT",
                 r["accuracy"], r["f1"], r["precision"], r["recall"], now]
            )

        # Insert bias_scorecard
        for model_name, r in summary.items():
            disparity = round(abs(r["real_accuracy"] - r["fake_accuracy"]), 4)
            cursor.execute(
                """INSERT INTO bias_scorecard
                       (model_name, dataset_name, label_bias,
                        fake_accuracy, real_accuracy, disparity_score,
                        evaluated_at)
                   VALUES (:1, :2, :3, :4, :5, :6, :7)""",
                [model_name, "ISOT", "FAKE vs REAL",
                 r["fake_accuracy"], r["real_accuracy"], disparity, now]
            )

        # Insert cross_dataset_performance (within ISOT)
        for model_name, r in summary.items():
            cursor.execute(
                """INSERT INTO cross_dataset_performance
                       (model_name, train_dataset, test_dataset,
                        accuracy, f1_score, evaluated_at)
                   VALUES (:1, :2, :3, :4, :5, :6)""",
                [model_name, "ISOT", "ISOT",
                 r["accuracy"], r["f1"], now]
            )

        conn.commit()
        cursor.close()
        conn.close()
        print("✅  Training results saved to Oracle.")

    except Exception as e:
        print(f"⚠️   Could not save to Oracle: {e}")
        print("    Run  python load_data.py  after Oracle is configured.")


if __name__ == "__main__":
    train_all()
