"""
model.py
========
Multi-model ML inference for FakeNewsDB.

Supported models (all loaded from models/ directory):
  NaiveBayes  -- nb.pkl      TF-IDF + MultinomialNB
  LogReg      -- lr.pkl      TF-IDF + LogisticRegression
  BiLSTM      -- bilstm.pkl  char-TF-IDF + SGDClassifier (gradient-descent proxy)
  BERT        -- bert.pkl    word n-gram TF-IDF + CalibratedLinearSVC (BERT proxy)

Public API:
  predict_news(text, model_name)
      -> { "prediction": "REAL" | "FAKE",
           "confidence": float (0-100),
           "model": model_name }

  available_models()
      -> list of loaded model names

Run train_models.py first to generate the .pkl files.
"""

import os
import re
import sys
import pickle

# ---- Paths ------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Mapping: UI name -> pkl filename (without extension)
MODEL_FILES = {
    "NaiveBayes": "nb",
    "LogReg":     "lr",
    "BiLSTM":     "bilstm",
    "BERT":       "bert",
}

DEFAULT_MODEL = "LogReg"


# ---- Text pre-processing ----------------------------------------------------
def preprocess(text: str) -> str:
    """
    Lightweight text cleaner -- must match train_models.py preprocessing exactly.
    Steps: lowercase -> strip URLs -> remove non-alphanumeric -> collapse whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---- Model loading ----------------------------------------------------------
_loaded_models = {}


def _load_model(name: str):
    """Load a single model from disk; raise FileNotFoundError if missing."""
    pkl_base = MODEL_FILES.get(name)
    if pkl_base is None:
        raise ValueError(
            "Unknown model '{}'. Valid options: {}".format(name, list(MODEL_FILES.keys()))
        )

    path = os.path.join(MODELS_DIR, "{}.pkl".format(pkl_base))
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Model file not found: {}\n"
            "Run 'python train_models.py' first.".format(path)
        )

    with open(path, "rb") as f:
        pipeline = pickle.load(f)

    print("  [model.py] Loaded {} from {}".format(name, path), flush=True)
    return pipeline


def _get_model(name: str):
    """Return cached model, loading from disk on first access."""
    if name not in _loaded_models:
        _loaded_models[name] = _load_model(name)
    return _loaded_models[name]


def _preload_all():
    """Pre-load all available models at startup (non-fatal if a file is missing)."""
    for name in MODEL_FILES:
        try:
            _get_model(name)
        except FileNotFoundError as e:
            print("  [model.py] WARNING: {}".format(e), file=sys.stderr)
        except Exception as e:
            print("  [model.py] WARNING: Could not load {}: {}".format(name, e), file=sys.stderr)


# Pre-load on import
_preload_all()


# ---- Public API -------------------------------------------------------------
def predict_news(text: str, model_name: str = DEFAULT_MODEL) -> dict:
    """
    Classify a news snippet as REAL or FAKE.

    Parameters
    ----------
    text       : str   Raw news text (title and/or body).
    model_name : str   One of NaiveBayes | LogReg | BiLSTM | BERT.

    Returns
    -------
    dict  { "prediction": "REAL" | "FAKE",
            "confidence": float (percentage 0-100),
            "model": str }
    """
    if not text or not text.strip():
        return {
            "prediction": "UNKNOWN",
            "confidence": 0.0,
            "model": model_name,
            "error": "Empty input text."
        }

    pipeline = _get_model(model_name)
    clean    = preprocess(text)

    if hasattr(pipeline, "predict_proba"):
        proba    = pipeline.predict_proba([clean])[0]
        pred_idx = int(proba.argmax())
        conf     = round(float(proba[pred_idx]) * 100, 1)
    else:
        pred_idx = int(pipeline.predict([clean])[0])
        conf     = 85.0  # default confidence when predict_proba unavailable

    label = "REAL" if pred_idx == 1 else "FAKE"
    return {"prediction": label, "confidence": conf, "model": model_name}


def predict_news_legacy(text: str) -> str:
    """Backward-compatible single-string return used by older code."""
    return predict_news(text, DEFAULT_MODEL)["prediction"]


def predict_with_confidence(text: str) -> dict:
    """Legacy helper -- uses the default model."""
    return predict_news(text, DEFAULT_MODEL)


def available_models() -> list:
    """Return list of model names that have been successfully loaded."""
    return [name for name in MODEL_FILES if name in _loaded_models]
