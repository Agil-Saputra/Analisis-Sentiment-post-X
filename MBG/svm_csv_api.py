import csv
import html
import io
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


MODEL_PATH = Path(__file__).with_name("svm_mbg_sentiment_model.joblib")
FRONTEND_PATH = Path(__file__).with_name("frontend_csv_example.html")

INDONESIAN_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "atau",
    "pada", "adalah", "dalam", "karena", "sebagai", "akan", "sudah", "jadi",
    "ada", "aja", "saja", "juga", "lebih", "masih", "lagi", "kok", "nih",
    "dong", "lah", "pun", "nya", "rt", "https", "http", "co", "t", "x", "com",
    "the", "a", "an", "of", "to", "for", "in", "on", "at", "by", "is", "are",
}

INDONESIAN_LABELS = {
    "positive": "positif",
    "neutral": "netral",
    "netral": "netral",
    "negative": "negatif",
}

SENTIMENT_ORDER = ["positive", "neutral", "negative"]

PUBLIC_SENTIMENT_LABELS = {
    "positive": "positive",
    "neutral": "neutral",
    "netral": "neutral",
    "negative": "negative",
}

app = FastAPI(title="SVM MBG Sentiment CSV API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_x_text(text: str) -> str:
    text = html.unescape(str(text))
    text = text.replace("�", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ_\s-]", " ", text)
    text = text.lower()
    tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ_]+", text)
    tokens = [
        token
        for token in tokens
        if len(token) > 2 and token not in INDONESIAN_STOPWORDS
    ]
    return " ".join(tokens)


@lru_cache(maxsize=1)
def load_svm_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "joblib is not installed. Run: python3 -m pip install -r requirements.txt"
        ) from exc

    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"]
    return bundle


def decode_csv_bytes(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="Could not decode CSV file.")


def read_csv_rows(file_bytes: bytes) -> tuple[list[str], list[dict[str, str]]]:
    text = decode_csv_bytes(file_bytes)
    reader = csv.DictReader(io.StringIO(text))

    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV must have a header row.")

    rows = [{key: value or "" for key, value in row.items()} for row in reader]
    if not rows:
        raise HTTPException(status_code=400, detail="CSV must contain at least one data row.")

    return reader.fieldnames, rows


def public_sentiment_label(label: str) -> str:
    return PUBLIC_SENTIMENT_LABELS.get(str(label).strip().lower(), str(label))


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "SVM MBG Sentiment CSV API",
        "dashboard_url": "/csv-dashboard",
        "predict_csv_url": "/api/sentiment/predict-csv",
        "docs_url": "/docs",
    }


@app.get("/csv-dashboard", response_class=HTMLResponse)
def csv_dashboard() -> str:
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="CSV dashboard file not found.")
    return FRONTEND_PATH.read_text(encoding="utf-8")


@app.post("/api/sentiment/predict-csv")
async def predict_csv(
    file: UploadFile = File(...),
    text_column: str = Form("text"),
) -> dict[str, Any]:
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

    try:
        model = load_svm_model()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    cleaned_texts: list[str] = []
    predicted_row_indexes: list[int] = []
    results: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        original_text = row.get(text_column, "").strip()
        result = dict(row)
        result["row_number"] = index
        result["clean_text"] = clean_x_text(original_text)
        result["predicted_sentiment"] = None
        result["predicted_sentiment_indonesia"] = None
        result["error"] = None

        if not original_text:
            result["error"] = f"Empty text in column '{text_column}'."
        elif not result["clean_text"]:
            result["error"] = "Text became empty after cleaning."
        else:
            cleaned_texts.append(result["clean_text"])
            predicted_row_indexes.append(len(results))

        results.append(result)

    if cleaned_texts:
        predictions = model.predict(cleaned_texts)
        for result_index, prediction in zip(predicted_row_indexes, predictions):
            prediction = public_sentiment_label(str(prediction))
            results[result_index]["predicted_sentiment"] = prediction
            results[result_index]["predicted_sentiment_indonesia"] = INDONESIAN_LABELS.get(
                prediction,
                prediction,
            )

    items_by_sentiment = {sentiment: [] for sentiment in SENTIMENT_ORDER}
    skipped_items = []

    for result in results:
        sentiment = result.get("predicted_sentiment")
        if sentiment in items_by_sentiment:
            items_by_sentiment[sentiment].append(result)
        else:
            skipped_items.append(result)

    summary = [
        {
            "sentiment": sentiment,
            "sentiment_indonesia": INDONESIAN_LABELS.get(sentiment, sentiment),
            "count": len(items_by_sentiment[sentiment]),
            "percent": round((len(items_by_sentiment[sentiment]) / len(cleaned_texts) * 100), 2)
            if cleaned_texts
            else 0,
        }
        for sentiment in SENTIMENT_ORDER
    ]

    return {
        "filename": file.filename,
        "text_column": text_column,
        "total_rows": len(results),
        "predicted_rows": len(cleaned_texts),
        "skipped_rows": len(results) - len(cleaned_texts),
        "summary": summary,
        "items_by_sentiment": items_by_sentiment,
        "skipped_items": skipped_items,
        "items": results,
    }
