# Run FastAPI Locally

This project runs a FastAPI service for CSV-based PSSI sentiment prediction.

## Requirements

- Python 3.10 or newer
- Terminal access on your local device
- The files below must stay in this folder:
  - `svm_csv_api.py`
  - `svm_pssi_sentiment_model.joblib`
  - `frontend_csv_example.html`
  - `requirements.txt`

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

After activation, your terminal prompt should show `(.venv)`.

## 3. Install Dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 4. Run The FastAPI Server

```bash
python3 -m uvicorn svm_csv_api:app --reload --host 127.0.0.1 --port 8000
```

If it starts successfully, the API will be available at:

```text
http://127.0.0.1:8000
```

If port `8000` is already in use, run it on another port:

```bash
python3 -m uvicorn svm_csv_api:app --reload --host 127.0.0.1 --port 8001
```

Then open:

```text
http://127.0.0.1:8001
```

## 5. Open The Dashboard

Open this URL in your browser:

```text
http://127.0.0.1:8000/csv-dashboard
```

Use the dashboard to upload a CSV file and choose the text column for sentiment prediction.

## 6. Open API Documentation

FastAPI provides interactive documentation automatically.

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

Upload a CSV file for prediction:

```bash
curl -X POST "http://127.0.0.1:8000/api/sentiment/predict-csv" \
  -F "file=@pssi_x_posts.csv" \
  -F "text_column=text"
```

If your CSV uses a different text column name, replace `text` with the correct column name.

## Troubleshooting

### `ModuleNotFoundError`

Install the dependencies again:

```bash
python3 -m pip install -r requirements.txt
```

### `Model file not found`

Make sure this file exists in the project folder:

```text
svm_pssi_sentiment_model.joblib
```

### `Column 'text' was not found in the CSV`

Your CSV does not have a column named `text`. Check the header row in your CSV and pass the correct column name in the dashboard or curl command.

### Stop The Server

Press `Ctrl + C` in the terminal where Uvicorn is running.
