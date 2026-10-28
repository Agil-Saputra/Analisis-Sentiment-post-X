"""
FastAPI – PSSI Dual-Model CSV Sentiment Dashboard
==================================================
Model 1 : svm_pssi_sentiment_model.joblib  (TF-IDF + LinearSVC, weak-lexicon labels)
Model 2 : svm_model.pkl  (SVC linear + SMOTE, IndoBERT labels)
         Requires companion tfidf_vectorizer.pkl.
         If that file is absent the bundle is rebuilt automatically from
         pssi_x_posts.csv and saved as svm_model2_bundle.joblib.
"""

import csv
import html
import io
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


# ── Paths ────────────────────────────────────────────────────────────────────
_DIR = Path(__file__).parent

MODEL1_PATH   = _DIR / "svm_pssi_sentiment_model.joblib"   # complete pipeline
MODEL2_SVC    = _DIR / "svm_model.pkl"                      # SVC only
MODEL2_TFIDF  = _DIR / "tfidf_vectorizer.pkl"               # companion (may be absent)
MODEL2_BUNDLE = _DIR / "svm_model2_bundle.joblib"           # auto-built cache
TRAINING_CSV  = _DIR / "pssi_x_posts.csv"
FRONTEND_PATH = _DIR / "frontend_csv_example.html"

# ── NLP helpers ──────────────────────────────────────────────────────────────
INDONESIAN_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "atau",
    "pada", "adalah", "dalam", "karena", "sebagai", "akan", "sudah", "jadi",
    "ada", "aja", "saja", "juga", "lebih", "masih", "lagi", "kok", "nih",
    "dong", "lah", "pun", "nya", "rt", "https", "http", "co", "t", "x", "com",
    "the", "a", "an", "of", "to", "for", "in", "on", "at", "by", "is", "are",
}

INDONESIAN_LABELS = {
    "positive": "positif",
    "neutral":  "netral",
    "netral":   "netral",
    "negative": "negatif",
}

PUBLIC_SENTIMENT_LABELS = {
    "positive": "positive",
    "neutral":  "netral",
    "netral":   "netral",
    "negative": "negative",
}

SENTIMENT_ORDER = ["positive", "netral", "negative"]


def clean_x_text(text: str) -> str:
    text = html.unescape(str(text))
    text = text.replace("_", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ\s-]", " ", text)
    text = text.lower()
    tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+", text)
    tokens = [t for t in tokens if len(t) > 2 and t not in INDONESIAN_STOPWORDS]
    return " ".join(tokens)


def public_sentiment_label(label: str) -> str:
    return PUBLIC_SENTIMENT_LABELS.get(str(label).strip().lower(), str(label))


# ── Model 1 loader ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_model1() -> Any:
    """Load svm_pssi_sentiment_model.joblib (complete TF-IDF + LinearSVC pipeline)."""
    if not MODEL1_PATH.exists():
        raise FileNotFoundError(f"Model 1 not found: {MODEL1_PATH}")
    import joblib
    bundle = joblib.load(MODEL1_PATH)
    return bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle


# ── Model 2 loader / builder ─────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_model2() -> Any:
    """
    Load svm_model.pkl wrapped in a full TF-IDF pipeline.

    Priority:
      1. svm_model2_bundle.joblib  (cached complete pipeline)
      2. svm_model.pkl + tfidf_vectorizer.pkl  (both present)
      3. Rebuild: TF-IDF(bigram) + SVC(SMOTE) from pssi_x_posts.csv,
         using Model 1 predictions as training labels → saved as bundle.
    """
    import joblib
    from sklearn.pipeline import Pipeline

    # 1 – original pkl + companion vectorizer (highest priority)
    #     If the user adds tfidf_vectorizer.pkl this will be used automatically
    #     (requires server restart to clear the lru_cache).
    if MODEL2_SVC.exists() and MODEL2_TFIDF.exists():
        print("[Model 2] Loading svm_model.pkl + tfidf_vectorizer.pkl")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svc   = joblib.load(MODEL2_SVC)
            tfidf = joblib.load(MODEL2_TFIDF)
        pipeline = Pipeline([("tfidf", tfidf), ("svc", svc)])
        joblib.dump(pipeline, MODEL2_BUNDLE)  # cache for future startups
        return pipeline

    # 2 – cached auto-built bundle
    if MODEL2_BUNDLE.exists():
        print("[Model 2] Loading cached bundle:", MODEL2_BUNDLE)
        return joblib.load(MODEL2_BUNDLE)

    # 3 – rebuild from training data
    print("[Model 2] tfidf_vectorizer.pkl not found → rebuilding from training data …")
    return _build_model2_bundle()


def _build_model2_bundle() -> Any:
    """
    Reconstruct a complete TF-IDF + SVC pipeline that mirrors the notebook's
    second model.  Model 1 predictions are used as training labels (pseudo-labels).
    """
    import joblib
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    if not TRAINING_CSV.exists():
        raise FileNotFoundError(
            "tfidf_vectorizer.pkl is absent and pssi_x_posts.csv was not found "
            "to rebuild Model 2 automatically.  "
            "Please place tfidf_vectorizer.pkl next to svm_model.pkl."
        )

    # ── load & filter CSV ────────────────────────────────────────────────────
    na_vals = ["NULL", "null", "None", "none", ""]
    df = pd.read_csv(
        TRAINING_CSV,
        encoding="utf-8-sig",
        encoding_errors="replace",
        na_values=na_vals,
        keep_default_na=True,
    )
    df["text"] = df["text"].fillna("").astype(str)

    if "lang" in df.columns:
        df = df[df["lang"].eq("in")].copy()
    if "is_retweet" in df.columns:
        df = df[df["is_retweet"].eq(0)].copy()
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")

    df = df[df["text"].str.strip().ne("")].copy()
    df["clean_text"] = df["text"].apply(clean_x_text)
    df = df[df["clean_text"].str.len().gt(0)].reset_index(drop=True)

    print(f"[Model 2] Training corpus: {len(df):,} samples")

    # ── generate pseudo-labels with Model 1 ──────────────────────────────────
    model1 = load_model1()
    raw_labels = model1.predict(df["clean_text"].tolist())
    df["label"] = [public_sentiment_label(str(lbl)) for lbl in raw_labels]
    print("[Model 2] Label distribution:\n", df["label"].value_counts().to_string())

    X = df["clean_text"].values
    y = df["label"].values

    # ── SMOTE oversampling ───────────────────────────────────────────────────
    try:
        from imblearn.over_sampling import SMOTE
        from sklearn.feature_extraction.text import TfidfVectorizer as _TV

        _tv = _TV(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        X_vec = _tv.fit_transform(X)

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_vec, y)
        print(f"[Model 2] After SMOTE: {X_res.shape[0]:,} samples")
        use_smote = True
    except Exception as exc:
        print(f"[Model 2] SMOTE skipped ({exc}); training on original data.")
        use_smote = False

    # ── build full pipeline (for inference no SMOTE needed) ──────────────────
    svc_cfg = dict(kernel="linear", C=1.0, random_state=42,
                   class_weight="balanced", probability=True)

    if use_smote:
        svc_smote = SVC(**svc_cfg)
        svc_smote.fit(X_res, y_res)

        # Wrap in pipeline using the same fitted TF-IDF
        pipeline = Pipeline([("tfidf", _tv), ("svc", svc_smote)])
    else:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10_000, ngram_range=(1, 2),
                sublinear_tf=True, min_df=2)),
            ("svc", SVC(**svc_cfg)),
        ])
        pipeline.fit(X, y)

    import joblib
    joblib.dump(pipeline, MODEL2_BUNDLE)
    print(f"[Model 2] Bundle saved → {MODEL2_BUNDLE}")
    return pipeline


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="PSSI Dual-Model Sentiment CSV API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── CSV helpers ───────────────────────────────────────────────────────────────
def decode_csv_bytes(file_bytes: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="Could not decode CSV file.")


def read_csv_rows(file_bytes: bytes) -> tuple[list[str], list[dict[str, str]]]:
    text = decode_csv_bytes(file_bytes)
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV must have a header row.")
    rows = [{k: v or "" for k, v in row.items()} for row in reader]
    if not rows:
        raise HTTPException(status_code=400, detail="CSV must contain at least one data row.")
    return list(reader.fieldnames), rows


# ── Core prediction logic ─────────────────────────────────────────────────────
def _run_predictions(
    model: Any,
    rows: list[dict[str, Any]],
    text_column: str,
) -> tuple[list[dict[str, Any]], list[str], list[int]]:
    """
    Clean texts, batch-predict with *model*, return (results, cleaned_texts, predicted_indexes).
    results has the same length as rows with extra keys injected.
    """
    cleaned_texts: list[str] = []
    predicted_indexes: list[int] = []
    results: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        original = row.get(text_column, "").strip()
        result = dict(row)
        result["row_number"] = index
        result["clean_text"] = clean_x_text(original)
        result["predicted_sentiment"] = None
        result["predicted_sentiment_indonesia"] = None
        result["error"] = None

        if not original:
            result["error"] = f"Empty text in column '{text_column}'."
        elif not result["clean_text"]:
            result["error"] = "Text became empty after cleaning."
        else:
            cleaned_texts.append(result["clean_text"])
            predicted_indexes.append(len(results))

        results.append(result)

    if cleaned_texts:
        predictions = model.predict(cleaned_texts)
        for res_idx, pred in zip(predicted_indexes, predictions):
            pred = public_sentiment_label(str(pred))
            results[res_idx]["predicted_sentiment"] = pred
            results[res_idx]["predicted_sentiment_indonesia"] = INDONESIAN_LABELS.get(pred, pred)

    return results, cleaned_texts, predicted_indexes


def _build_response_payload(
    results: list[dict[str, Any]],
    cleaned_texts: list[str],
    filename: str,
    text_column: str,
) -> dict[str, Any]:
    """Group results and compute summary statistics."""
    items_by_sentiment: dict[str, list] = {s: [] for s in SENTIMENT_ORDER}
    skipped_items: list[dict] = []

    for result in results:
        sentiment = result.get("predicted_sentiment")
        if sentiment in items_by_sentiment:
            items_by_sentiment[sentiment].append(result)
        else:
            skipped_items.append(result)

    total_predicted = len(cleaned_texts)
    summary = [
        {
            "sentiment": s,
            "sentiment_indonesia": INDONESIAN_LABELS.get(s, s),
            "count": len(items_by_sentiment[s]),
            "percent": round(len(items_by_sentiment[s]) / total_predicted * 100, 2)
            if total_predicted else 0,
        }
        for s in SENTIMENT_ORDER
    ]

    return {
        "filename": filename,
        "text_column": text_column,
        "total_rows": len(results),
        "predicted_rows": total_predicted,
        "skipped_rows": len(results) - total_predicted,
        "summary": summary,
        "items_by_sentiment": items_by_sentiment,
        "skipped_items": skipped_items,
        "items": results,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "PSSI Dual-Model Sentiment CSV API",
        "dashboard_url": "/csv-dashboard",
        "predict_csv_url": "/api/sentiment/predict-csv",
        "docs_url": "/docs",
    }


@app.get("/csv-dashboard", response_class=HTMLResponse)
def csv_dashboard() -> str:
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Dashboard HTML file not found.")
    return FRONTEND_PATH.read_text(encoding="utf-8")


@app.post("/api/sentiment/predict-csv")
async def predict_csv(
    file: UploadFile = File(...),
    text_column: str = Form("text"),
) -> dict[str, Any]:
    """
    Predict sentiment for every row in the uploaded CSV using **both** models.

    Returns:
      - model1: predictions from svm_pssi_sentiment_model.joblib
      - model2: predictions from svm_model.pkl (or auto-rebuilt bundle)
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")

    file_bytes = await file.read()
    columns, rows = read_csv_rows(file_bytes)

    if text_column not in columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Column '{text_column}' was not found in the CSV.",
                "available_columns": columns,
            },
        )

    # Load both models (cached after first call)
    try:
        model1 = load_model1()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Model 1: {exc}") from exc

    try:
        model2 = load_model2()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Model 2: {exc}") from exc

    # Run predictions
    results1, cleaned1, _ = _run_predictions(model1, rows, text_column)
    results2, cleaned2, _ = _run_predictions(model2, rows, text_column)

    return {
        "filename": file.filename,
        "text_column": text_column,
        "model1": _build_response_payload(results1, cleaned1, file.filename, text_column),
        "model2": _build_response_payload(results2, cleaned2, file.filename, text_column),
    }
