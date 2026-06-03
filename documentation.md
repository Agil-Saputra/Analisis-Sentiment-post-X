# PSSI SVM Sentiment CSV API - Local Setup

This project runs a local FastAPI service that accepts a CSV file, predicts sentiment with the saved SVM model, returns percentage summaries, and groups all rows by sentiment.

## 1. Project Files

Important files:

| File | Purpose |
| --- | --- |
| `svm_csv_api.py` | FastAPI backend for CSV upload and sentiment prediction. |
| `frontend_csv_example.html` | Local frontend dashboard with CSV upload, chart, summary cards, and grouped tables. |
| `svm_pssi_sentiment_model.joblib` | Saved SVM model from `SVM_PSSI_SENTIMENT.ipynb`. |
| `requirements.txt` | Python dependencies. |
| `pssi_x_posts.csv` | Example CSV data. |

## 2. How Everything Connects

The project has three main parts:

```text
Notebook / Joblib Model -> FastAPI Backend -> Frontend Dashboard
```

Simple explanation:

1. `SVM_PSSI_SENTIMENT.ipynb` trains the sentiment model.
2. The notebook saves the trained model into `svm_pssi_sentiment_model.joblib`.
3. `svm_csv_api.py` loads that `.joblib` model with `joblib.load()`.
4. The frontend page `frontend_csv_example.html` sends a CSV file to FastAPI.
5. FastAPI reads the CSV, cleans the text, sends the clean text to the SVM model, and receives sentiment predictions.
6. FastAPI returns JSON containing sentiment percentage summary and grouped rows.
7. The frontend uses that JSON to draw a chart, summary cards, and tables.

Data flow:

```text
User uploads CSV
       |
       v
Frontend creates FormData
       |
       v
POST /api/sentiment/predict-csv
       |
       v
FastAPI reads CSV rows
       |
       v
FastAPI cleans text using clean_x_text()
       |
       v
SVM model predicts sentiment
       |
       v
FastAPI groups result by positive, netral, negative
       |
       v
Frontend displays chart and grouped tables
```

## 3. How The Joblib Model Connects To FastAPI

The file `svm_pssi_sentiment_model.joblib` is the saved machine learning model.

It was created in the notebook using code like this:

```python
joblib.dump({
    "model": best_model,
    "labels": list(best_model.named_steps["svm"].classes_),
    "label_source": LABEL_SOURCE,
}, "svm_pssi_sentiment_model.joblib")
```

Inside `svm_csv_api.py`, the model path is defined here:

```python
MODEL_PATH = Path(__file__).with_name("svm_pssi_sentiment_model.joblib")
```

This means FastAPI expects the model file to be in the same folder as `svm_csv_api.py`.

The model is loaded by this function:

```python
@lru_cache(maxsize=1)
def load_svm_model() -> Any:
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"]
    return bundle
```

What this does:

- `joblib.load(MODEL_PATH)` opens the `.joblib` file.
- The notebook saved a dictionary that contains `"model"`.
- The API takes `bundle["model"]`.
- `@lru_cache(maxsize=1)` makes the model load only once, then reuse the same model for the next requests.

This is important because loading a model every time a user uploads CSV would be slower.

## 4. How FastAPI Uses The Model

The main API endpoint is:

```python
@app.post("/api/sentiment/predict-csv")
async def predict_csv(...)
```

This endpoint receives two form fields:

```text
file        = uploaded CSV file
text_column = column name that contains the text
```

The endpoint does this step by step:

1. Check that the uploaded file is a `.csv`.
2. Read the uploaded file bytes.
3. Convert the CSV into rows using `csv.DictReader`.
4. Check that the selected text column exists.
5. Load the SVM model using `load_svm_model()`.
6. Clean each text using `clean_x_text()`.
7. Predict sentiment using:

```python
predictions = model.predict(cleaned_texts)
```

8. Add prediction result to each row.
9. Count the result for `positive`, `netral`, and `negative`.
10. Calculate percentage for each sentiment.
11. Group all rows into `items_by_sentiment`.
12. Return the final JSON response to the frontend.

Example model prediction flow:

```python
cleaned_texts = [
    "pssi makin bagus rapi",
    "keputusan pssi bikin kecewa",
]

predictions = model.predict(cleaned_texts)
```

Possible result:

```python
["positive", "negative"]
```

The model originally returns:

```text
positive
neutral
negative
```

The API converts `neutral` into `netral` for the public response:

```python
PUBLIC_SENTIMENT_LABELS = {
    "positive": "positive",
    "neutral": "netral",
    "netral": "netral",
    "negative": "negative",
}
```

So the frontend always receives:

```text
positive
netral
negative
```

## 5. How FastAPI Connects To The Frontend

The frontend is the file:

```text
frontend_csv_example.html
```

FastAPI serves that file through this route:

```python
@app.get("/csv-dashboard", response_class=HTMLResponse)
def csv_dashboard() -> str:
    return FRONTEND_PATH.read_text(encoding="utf-8")
```

So when you open:

```text
http://127.0.0.1:8002/csv-dashboard
```

FastAPI sends the HTML page to your browser.

Inside the frontend, this line defines where the CSV should be sent:

```javascript
const API_URL = '/api/sentiment/predict-csv';
```

When the user clicks `Analyze CSV`, the frontend creates a `FormData` payload:

```javascript
const formData = new FormData();

formData.append('file', file);
formData.append('text_column', textColumn);
```

Then it sends that payload to FastAPI:

```javascript
const response = await fetch(API_URL, {
  method: 'POST',
  body: formData,
});
```

FastAPI processes the CSV and returns JSON.

The frontend then uses:

```javascript
renderResults(data, textColumn);
```

That function calls:

```javascript
renderChart(data.summary);
renderLegend(data.summary);
renderSummary(data.summary);
renderGroupedTables(data.items_by_sentiment, textColumn);
```

Meaning:

- `renderChart()` draws the sentiment percentage chart.
- `renderSummary()` shows the percentage and count cards.
- `renderGroupedTables()` shows all rows separated by sentiment.

## 6. Simple End-To-End Example

Suppose the uploaded CSV is:

```csv
id,text
1,PSSI sekarang makin bagus dan rapi
2,Keputusan PSSI bikin kecewa
3,PSSI akan mengumumkan jadwal pertandingan
```

The frontend sends this CSV to:

```text
POST /api/sentiment/predict-csv
```

FastAPI reads the `text` column:

```text
PSSI sekarang makin bagus dan rapi
Keputusan PSSI bikin kecewa
PSSI akan mengumumkan jadwal pertandingan
```

FastAPI cleans the text:

```text
pssi sekarang makin bagus rapi
keputusan pssi bikin kecewa
pssi mengumumkan jadwal pertandingan
```

The SVM model predicts:

```text
positive
negative
netral
```

FastAPI returns a response like:

```json
{
  "summary": [
    {
      "sentiment": "positive",
      "count": 1,
      "percent": 33.33
    },
    {
      "sentiment": "netral",
      "count": 1,
      "percent": 33.33
    },
    {
      "sentiment": "negative",
      "count": 1,
      "percent": 33.33
    }
  ],
  "items_by_sentiment": {
    "positive": [
      {
        "id": "1",
        "text": "PSSI sekarang makin bagus dan rapi",
        "clean_text": "pssi sekarang makin bagus rapi",
        "predicted_sentiment": "positive"
      }
    ],
    "netral": [
      {
        "id": "3",
        "text": "PSSI akan mengumumkan jadwal pertandingan",
        "clean_text": "pssi mengumumkan jadwal pertandingan",
        "predicted_sentiment": "netral"
      }
    ],
    "negative": [
      {
        "id": "2",
        "text": "Keputusan PSSI bikin kecewa",
        "clean_text": "keputusan pssi bikin kecewa",
        "predicted_sentiment": "negative"
      }
    ]
  }
}
```

The frontend uses this response to show:

- one chart with three sentiment percentages
- one card for each sentiment
- one table for positive rows
- one table for netral rows
- one table for negative rows

## 7. Requirements

Use Python 3.10 or newer.

Check your Python version:

```bash
python3 --version
```

The model file must exist in the same folder as `svm_csv_api.py`:

```text
svm_pssi_sentiment_model.joblib
```

## 8. Open The Project Folder

Run this in Terminal:

```bash
cd /Users/agilsaputra/datasets
```

## 9. Install Dependencies

Install all required packages:

```bash
python3 -m pip install -r requirements.txt
```

The important packages for this API are:

```text
fastapi
uvicorn
joblib
scikit-learn
python-multipart
```

`python-multipart` is required because the frontend sends the CSV as `multipart/form-data`.

## 10. Run The Local API

Start the server:

```bash
python3 -m uvicorn svm_csv_api:app --host 127.0.0.1 --port 8002
```

If it works, you should see something like:

```text
Uvicorn running on http://127.0.0.1:8002
```

Keep this Terminal window open while using the API.

## 11. Open The Frontend Dashboard

Open this URL in your browser:

```text
http://127.0.0.1:8002/csv-dashboard
```

The dashboard lets you:

- upload a CSV file
- choose the text column name
- see a sentiment percentage chart
- see summary cards for `positive`, `netral`, and `negative`
- see all rows grouped by sentiment

## 12. CSV Format

Your CSV must have a header row and one column containing the text to analyze.

Example:

```csv
id,text
1,PSSI sekarang makin bagus dan rapi
2,Keputusan PSSI bikin kecewa
3,PSSI akan mengumumkan jadwal pertandingan
```

By default, the API reads the column named:

```text
text
```

If your CSV uses another column name, such as `tweet`, `content`, or `post`, type that column name in the frontend input.

## 13. API Endpoint

CSV prediction endpoint:

```http
POST /api/sentiment/predict-csv
```

Full local URL:

```text
http://127.0.0.1:8002/api/sentiment/predict-csv
```

Payload type:

```text
multipart/form-data
```

Form fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `file` | CSV file | Yes | The CSV file to analyze. |
| `text_column` | string | No | Column name that contains the text. Default is `text`. |

## 14. Test With Curl

Example using the included CSV:

```bash
curl -X POST "http://127.0.0.1:8002/api/sentiment/predict-csv" \
  -F "file=@pssi_x_posts.csv" \
  -F "text_column=text"
```

To save the response to a JSON file:

```bash
curl -X POST "http://127.0.0.1:8002/api/sentiment/predict-csv" \
  -F "file=@pssi_x_posts.csv" \
  -F "text_column=text" \
  -o prediction_result.json
```

## 15. Response Format

The API returns:

```json
{
  "filename": "pssi_x_posts.csv",
  "text_column": "text",
  "total_rows": 2694,
  "predicted_rows": 2675,
  "skipped_rows": 19,
  "summary": [
    {
      "sentiment": "positive",
      "sentiment_indonesia": "positif",
      "count": 295,
      "percent": 11.03
    },
    {
      "sentiment": "netral",
      "sentiment_indonesia": "netral",
      "count": 1763,
      "percent": 65.91
    },
    {
      "sentiment": "negative",
      "sentiment_indonesia": "negatif",
      "count": 617,
      "percent": 23.07
    }
  ],
  "items_by_sentiment": {
    "positive": [],
    "netral": [],
    "negative": []
  },
  "skipped_items": [],
  "items": []
}
```

Main response fields:

| Field | Meaning |
| --- | --- |
| `total_rows` | Total rows found in the uploaded CSV. |
| `predicted_rows` | Rows successfully predicted by the model. |
| `skipped_rows` | Rows skipped because text was empty or became empty after cleaning. |
| `summary` | Count and percentage for each sentiment. |
| `items_by_sentiment.positive` | All rows predicted as positive. |
| `items_by_sentiment.netral` | All rows predicted as netral. |
| `items_by_sentiment.negative` | All rows predicted as negative. |
| `skipped_items` | Rows that could not be predicted. |
| `items` | All rows, including predicted and skipped rows. |

The percentage is calculated from `predicted_rows`, not `total_rows`.

## 16. Frontend Fetch Example

Use `FormData` from the browser:

```javascript
const formData = new FormData();

formData.append("file", csvFile);
formData.append("text_column", "text");

const response = await fetch("http://127.0.0.1:8002/api/sentiment/predict-csv", {
  method: "POST",
  body: formData,
});

const data = await response.json();

console.log(data.summary);
console.log(data.items_by_sentiment.positive);
console.log(data.items_by_sentiment.netral);
console.log(data.items_by_sentiment.negative);
```

Do not manually set the `Content-Type` header. The browser will set the correct `multipart/form-data` boundary automatically.

## 17. Interactive API Docs

FastAPI also provides automatic API documentation:

```text
http://127.0.0.1:8002/docs
```

You can test the CSV upload endpoint from that page.

## 18. Stop The Server

In the Terminal window running Uvicorn, press:

```text
Control + C
```

## 19. Common Problems

### Port already in use

If port `8002` is already used, run on another port:

```bash
python3 -m uvicorn svm_csv_api:app --host 127.0.0.1 --port 8003
```

Then open:

```text
http://127.0.0.1:8003/csv-dashboard
```

### Column not found

If the API says the text column was not found, check the CSV header.

Example CSV header:

```csv
id,text,created_at
```

For this CSV, use:

```text
text
```

### Model file not found

Make sure this file exists:

```text
/Users/agilsaputra/datasets/svm_pssi_sentiment_model.joblib
```

If it does not exist, rerun the notebook cell that saves the model.
