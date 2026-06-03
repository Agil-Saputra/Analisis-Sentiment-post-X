# PSSI Sentiment API Documentation

This API serves sentiment analysis results from `pssi_x_posts.json`. It is built with FastAPI and is designed to be consumed by a frontend application such as React, Vue, Next.js, plain HTML, or a mobile app.

## Project Files

- `main.py`: FastAPI application.
- `export_keras_model.py`: Trains and exports `pssi_sentiment_model.keras`.
- `pssi_sentiment_model.keras`: Exported Keras sentiment model created by the export script.
- `pssi_sentiment_labels.json`: Label order used by the exported Keras model.
- `pssi_x_posts.json`: Labeled dataset used by the API.
- `frontend_example.html`: Simple frontend dashboard example.
- `requirements.txt`: Python dependencies.
- `pssi_sentiment_analysis.ipynb`: Colab notebook for notebook-based analysis.

## Requirements

- Python 3.10 or newer.
- `pssi_x_posts.json` must be in the same folder as `main.py`.
- TensorFlow is required to export and run the `.keras` prediction endpoint.
- Each post should already contain a `sentiment` field.

Allowed sentiment values:

- `negative`
- `netral`
- `positive`

## Installation

Open a terminal in this folder:

```bash
cd /Users/agilsaputra/datasets
```

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Export The Keras Model

The notebook model in `SVM_PSSI_SENTIMENT.ipynb` is an SVM saved as `.joblib`, so it cannot be directly converted into a `.keras` model. To produce a real Keras model, run the included TensorFlow exporter. It trains a text classifier from `pssi_x_posts.json` and saves a model that accepts raw text strings.

```bash
python3 export_keras_model.py
```

Expected output files:

```text
pssi_sentiment_model.keras
pssi_sentiment_labels.json
```

Run this export before calling the prediction endpoint.

## Run The API

Start the FastAPI server:

```bash
python3 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Base URL:

```text
http://127.0.0.1:8000
```

If port `8000` is already used, run it on another port:

```bash
python3 -m uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

Then use:

```text
http://127.0.0.1:8001
```

## Interactive API Docs

FastAPI automatically provides interactive API documentation.

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

Use these pages to test endpoints directly in the browser.

## CORS

The API currently allows requests from any frontend origin:

```python
allow_origins=["*"]
```

This is convenient for development. For production, replace `*` with your frontend URL, for example:

```python
allow_origins=["https://your-frontend-domain.com"]
```

## Endpoints

### 1. Root

Returns basic API information and route paths.

```http
GET /
```

Example request:

```bash
curl http://127.0.0.1:8000/
```

Example response:

```json
{
  "message": "PSSI Sentiment API",
  "dashboard_url": "/dashboard",
  "summary_url": "/api/sentiment/summary",
  "posts_url": "/api/posts",
  "predict_url": "/api/sentiment/predict"
}
```

### 2. Dashboard

Returns the example frontend HTML page.

```http
GET /dashboard
```

Open this in your browser:

```text
http://127.0.0.1:8000/dashboard
```

This page fetches data from:

```text
http://localhost:8000/api/sentiment/summary
```

If you run the API on a different port, update `API_URL` inside `frontend_example.html`.

### 3. Sentiment Summary

Returns sentiment count and percentage.

```http
GET /api/sentiment/summary
```

Example request:

```bash
curl http://127.0.0.1:8000/api/sentiment/summary
```

Example response:

```json
{
  "total_posts": 2694,
  "summary": [
    {
      "sentiment": "negative",
      "count": 1039,
      "percent": 38.57
    },
    {
      "sentiment": "netral",
      "count": 1496,
      "percent": 55.53
    },
    {
      "sentiment": "positive",
      "count": 159,
      "percent": 5.9
    }
  ]
}
```

Response fields:

| Field | Type | Description |
| --- | --- | --- |
| `total_posts` | number | Total posts with valid sentiment labels. |
| `summary` | array | List of sentiment summary objects. |
| `summary[].sentiment` | string | Sentiment label: `negative`, `netral`, or `positive`. |
| `summary[].count` | number | Number of posts for that sentiment. |
| `summary[].percent` | number | Percentage of total posts for that sentiment. |

Percentage formula:

```text
count / total_posts * 100
```

The result is rounded to two decimal places.

### 4. Predict Sentiment

Runs the exported Keras model on new text.

```http
POST /api/sentiment/predict
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/api/sentiment/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts":["PSSI harus lebih transparan soal keputusan wasit.","Selamat untuk timnas Indonesia!"]}'
```

Example response:

```json
{
  "predictions": [
    {
      "text": "PSSI harus lebih transparan soal keputusan wasit.",
      "sentiment": "negative",
      "confidence": 0.812345,
      "probabilities": {
        "negative": 0.812345,
        "netral": 0.145678,
        "positive": 0.041977
      }
    }
  ]
}
```

If `pssi_sentiment_model.keras` does not exist yet, run:

```bash
python3 export_keras_model.py
```

### 5. Posts

Returns post data from the dataset. Supports pagination and optional sentiment filtering.

```http
GET /api/posts
```

Query parameters:

| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `sentiment` | string | No | `null` | Filter by `negative`, `netral`, or `positive`. |
| `limit` | number | No | `50` | Number of rows returned. Minimum `1`, maximum `500`. |
| `offset` | number | No | `0` | Number of rows to skip. Useful for pagination. |

Example request for first 10 posts:

```bash
curl "http://127.0.0.1:8000/api/posts?limit=10"
```

Example request for positive posts:

```bash
curl "http://127.0.0.1:8000/api/posts?sentiment=positive&limit=10"
```

Example request for pagination:

```bash
curl "http://127.0.0.1:8000/api/posts?limit=10&offset=20"
```

Example response:

```json
{
  "total": 159,
  "limit": 1,
  "offset": 0,
  "items": [
    {
      "id": "2037907123905196493",
      "text": "RT @Aralie_JKT48: TERIMA KASIH PSSI AWARD ...",
      "author_username": "heikllskamichie",
      "created_at": "2026-03-28 14:58:15",
      "created_at_date": "2026-03-28",
      "like_count": 0,
      "retweet_count": 2954,
      "reply_count": 0,
      "quote_count": 0,
      "impression_count": 0,
      "bookmark_count": 0,
      "lang": "in",
      "is_retweet": true,
      "sentiment": "positive"
    }
  ]
}
```

Response fields:

| Field | Type | Description |
| --- | --- | --- |
| `total` | number | Total rows after applying the sentiment filter. |
| `limit` | number | Maximum rows requested. |
| `offset` | number | Number of skipped rows. |
| `items` | array | Post objects from the dataset. |

The fields inside `items` come directly from `pssi_x_posts.json`.

## Frontend Usage

### Plain JavaScript

```html
<script>
  async function getSentimentSummary() {
    const response = await fetch("http://127.0.0.1:8000/api/sentiment/summary");

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    console.log(data.summary);
  }

  getSentimentSummary();
</script>
```

### React Example

```jsx
import { useEffect, useState } from "react";

export default function SentimentSummary() {
  const [summary, setSummary] = useState([]);
  const [totalPosts, setTotalPosts] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadSummary() {
      try {
        const response = await fetch("http://127.0.0.1:8000/api/sentiment/summary");

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        setSummary(data.summary);
        setTotalPosts(data.total_posts);
      } catch (err) {
        setError(err.message);
      }
    }

    loadSummary();
  }, []);

  if (error) return <p>{error}</p>;

  return (
    <section>
      <h1>PSSI Sentiment</h1>
      <p>Total posts: {totalPosts}</p>

      {summary.map((item) => (
        <article key={item.sentiment}>
          <h2>{item.sentiment}</h2>
          <p>{item.percent}%</p>
          <p>{item.count} posts</p>
        </article>
      ))}
    </section>
  );
}
```

### Fetch Filtered Posts

```js
async function getPositivePosts() {
  const response = await fetch(
    "http://127.0.0.1:8000/api/posts?sentiment=positive&limit=10"
  );

  const data = await response.json();
  return data.items;
}
```

## Error Responses

### Dataset Missing

If `pssi_x_posts.json` is missing:

```json
{
  "detail": "Dataset not found: /path/to/pssi_x_posts.json"
}
```

Status code:

```text
500 Internal Server Error
```

### Invalid Sentiment Query

If `sentiment` is not one of `negative`, `netral`, or `positive`:

```http
GET /api/posts?sentiment=happy
```

FastAPI returns:

```text
422 Unprocessable Entity
```

Use one of:

```text
negative
netral
positive
```

### Invalid Pagination

Examples:

```http
GET /api/posts?limit=0
GET /api/posts?limit=1000
GET /api/posts?offset=-1
```

FastAPI returns:

```text
422 Unprocessable Entity
```

Valid values:

- `limit`: `1` to `500`
- `offset`: `0` or greater

## Pagination Pattern

Use `limit` and `offset` from the frontend.

Example:

```js
let page = 0;
const limit = 20;

async function loadPage(page) {
  const offset = page * limit;
  const response = await fetch(
    `http://127.0.0.1:8000/api/posts?limit=${limit}&offset=${offset}`
  );

  return response.json();
}
```

Page examples:

| Page | Limit | Offset |
| --- | --- | --- |
| 1 | 20 | 0 |
| 2 | 20 | 20 |
| 3 | 20 | 40 |

## Deployment Notes

For local development:

```bash
python3 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

For deployment, use:

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Important production notes:

- Restrict CORS to your real frontend domain.
- Keep `pssi_x_posts.json` available on the server.
- Do not use `--reload` in production.
- Put the API behind HTTPS when used outside local development.

## Current Dataset Result

Based on the current `pssi_x_posts.json`:

| Sentiment | Count | Percent |
| --- | ---: | ---: |
| negative | 1039 | 38.57% |
| netral | 1496 | 55.53% |
| positive | 159 | 5.90% |

Total posts:

```text
2694
```

## Quick Test Checklist

After starting the API, open these URLs:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/api/sentiment/summary
http://127.0.0.1:8000/api/posts?sentiment=negative&limit=5
http://127.0.0.1:8000/dashboard
http://127.0.0.1:8000/docs
```

If each URL loads, the API is ready for frontend consumption.
