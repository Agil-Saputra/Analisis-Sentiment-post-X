# Run MBG Sentiment FastAPI Locally

This guide explains how to run the MBG sentiment CSV API from the `MBG` folder.

## Requirements

- Python 3.10 or newer
- Terminal access
- These files must exist in the `MBG` folder:
  - `svm_csv_api.py`
  - `svm_mbg_sentiment_model.joblib`
  - `frontend_csv_example.html`
  - `mbg_x_posts.csv` if you want to test with the included dataset

## 1. Open The Project Folder

```bash
cd /Users/agilsaputra/datasets
```

## 2. Create A Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install pandas matplotlib seaborn httpx
```

The API itself mainly needs FastAPI, Uvicorn, Joblib, scikit-learn, and python-multipart. The extra packages are useful if you also run the notebook or local tests.

## 4. Run The API

Run Uvicorn from inside the `MBG` folder:

```bash
cd /Users/agilsaputra/datasets/MBG
python3 -m uvicorn svm_csv_api:app --reload --host 127.0.0.1 --port 8000
```

If port `8000` is already in use, choose another port:

```bash
python3 -m uvicorn svm_csv_api:app --reload --host 127.0.0.1 --port 8001
```

## 5. Open The Dashboard

Open this URL in your browser:

```text
http://127.0.0.1:8000/csv-dashboard
```

Upload a CSV file and set the text column name. For `mbg_x_posts.csv`, use:

```text
text
```

## 6. Open API Docs

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

## 7. Test With Curl

Check that the API is running:

```bash
curl http://127.0.0.1:8000/
```

Predict sentiment from the included MBG CSV:

```bash
curl -X POST "http://127.0.0.1:8000/api/sentiment/predict-csv" \
  -F "file=@mbg_x_posts.csv" \
  -F "text_column=text"
```

If your CSV uses a different text column, replace `text` with the correct column name.

## Main Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | API status and route links |
| `GET` | `/csv-dashboard` | Browser dashboard for CSV upload |
| `POST` | `/api/sentiment/predict-csv` | Predict sentiment for uploaded CSV rows |
| `GET` | `/docs` | Swagger API documentation |
| `GET` | `/redoc` | ReDoc API documentation |

## Response Labels

The API returns these sentiment labels:

- `positive`
- `neutral`
- `negative`

It also returns Indonesian labels in `predicted_sentiment_indonesia`:

- `positif`
- `netral`
- `negatif`

## Troubleshooting

### `ModuleNotFoundError`

Install dependencies again:

```bash
cd /Users/agilsaputra/datasets
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### `Model file not found`

Make sure the model file exists:

```text
/Users/agilsaputra/datasets/MBG/svm_mbg_sentiment_model.joblib
```

### `Column 'text' was not found in the CSV`

Your CSV does not have a column named `text`. Check the header row and pass the correct column name in the dashboard or curl command.

### Stop The Server

Press `Ctrl + C` in the terminal where Uvicorn is running.
