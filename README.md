Absolutely! Here's your revised `README.md` with a **handcrafted, human-style formatting** using `#`, `*`, and other common markdown conventions. It maintains a **developer-authored** tone—**clear, clean, and not overly polished like AI-generated content**—but still looks professional and readable.

---

# 🧠 Post Similarity & Tag Generation Microservice

A microservice using FastAPI that detects duplicate questions and auto-generates tags using sentence embeddings. Runs fully offline, Dockerized, and easy to plug into educational or real estate platforms.

---

## 📌 Overview

This project uses NLP to:

* Check for similar/duplicate posts using cosine similarity over embeddings
* Generate tags automatically using a trained tag classifier

Useful for:

* 👨‍🏫 Student portals with teacher-managed Q\&A
* 🏠 Real estate sites needing tag suggestions for listings

---

## 🧰 Tech Stack

* Python 3.10 (slim)
* FastAPI
* Sentence Transformers
* FAISS
* Docker
* Pytest

---

## 📁 Project Structure

```
Microservice_turtil/
├── .dockerignore            # Ignore files/folders in Docker image
├── .gitignore               # Ignore files/folders in Git
├── Dockerfile               # Docker build steps
├── main.py                  # FastAPI main app
├── requirements.txt         # All Python dependencies

├── frontend/
│   └── index.html           # Simple static UI on localhost:8000

├── Similairty_check/
│   ├── config.json          # Thresholds and other configs
│   ├── faiss_index.index    # FAISS index for similarity search
│   ├── fastapi_search.py    # API for duplicate checking
│   ├── post_mapping.pkl     # FAISS ID to post ID mapping

├── Tag_generator/
│   ├── data/                       # Raw or preprocessed data
│   ├── models/                     # Trained models (.bin, .pkl)
│   ├── src/
│   │   ├── data_preprocessing.py  # Tokenizing, cleaning, etc.
│   │   ├── model.py               # BERT-based classifier
│   │   └── train.py               # Training logic
│   ├── run_training.py            # Triggers model training
│   ├── infere.py                  # Manual inference script
│   └── fastapi_tag.py             # API to get tags from text
```

---

## 🔧 Installation

```bash
git clone https://github.com/Rahulgillella22/turtil_internproject

# (Optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### ⚙️ One-Time Setup (Must Do)

> GitHub has a 100MB limit, so big models are downloaded via scripts.

* Go to `Similairty_check/`
  Run: `python script_downloadmodel.py`
  ➜ Downloads `allMiniLM-L6-v2` into models folder

* Then go to `Tag_generator/`
  Run: `python run_training.py`
  ➜ Generates `.bin` and `.pkl` model files

---

## 🚀 API Endpoints

### `POST /check-duplicate`

Checks if a post is similar to any existing one.

**Request:**

```json
{
  "user_id": "user_123",
  "post_title": "How to find duplicates in an array?",
  "post_body": "I need an efficient algorithm to find repeated numbers."
}
```

**Response:**

```json
{
  "is_duplicate": true,
  "similar_post_id": "post_003",
  "similarity_score": 0.91
}
```

---

### `POST /generate-tags`

Returns relevant tags based on the input text.

**Request:**

```json
{
  "text": "How can I understand dynamic programming using Python?"
}
```

**Response:**

```json
{
  "predicted_tags": ["dynamic programming", "python", "dp basics"]
}
```

---

##  Testing

* Used `pytest` for backend modules
* 8 test cases covering both duplicate detection and tag generation
* All pass offline ✅

To run tests:

```bash
pytest
```

---

## 🐳 Docker Setup

**Build the Docker image:**

```bash
docker build -t microservice-turtil .
```

**Run the container:**

```bash
docker run -p 8000:8000 microservice-turtil
```

> Static frontend available at: [http://localhost:8000](http://localhost:8000)
> API Docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

---

