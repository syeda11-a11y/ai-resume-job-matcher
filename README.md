# AI Resume ↔ Job Matcher

A beginner-friendly AI web app that compares a resume with a job description and produces:

- **Match score** (0–100) combining TF-IDF cosine similarity and keyword coverage
- **TF-IDF cosine similarity** between resume and job description text
- **JD keyword coverage** — percentage of top JD keywords found in your resume
- **Table of top JD keywords** ranked by frequency
- **Table of missing keywords** — top JD keywords not found in your resume
- **Suggestions** to help improve your resume for the role

100% offline — no external APIs required.

---

## Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/syeda11-a11y/ai-resume-job-matcher.git
cd ai-resume-job-matcher
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser automatically.

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
ai-resume-job-matcher/
├── app.py               # Streamlit web UI
├── matcher.py           # Text processing, TF-IDF, scoring logic
├── requirements.txt     # Pinned dependencies
├── tests/
│   └── test_matcher.py  # pytest test suite
└── .github/
    └── workflows/
        └── ci.yml       # GitHub Actions CI (runs tests on push/PR)
```

---

## How It Works

1. **Normalize** — lowercase, remove punctuation, collapse whitespace
2. **Tokenize** — split into words, filter by minimum length and stopwords
3. **Keyword counts** — count token frequencies per document
4. **TF-IDF cosine similarity** — scikit-learn `TfidfVectorizer` (1–2 grams) + `cosine_similarity`
5. **Coverage** — percentage of top JD keywords present in resume
6. **Score** — `similarity × 60 + coverage × 0.40` → clamped to 0–100

---

## UI Controls

| Slider | What it does |
|--------|-------------|
| Number of keywords to show | `top_k` — how many JD keywords to display |
| Minimum keyword length | Filters out very short tokens |
| Strong-match threshold | Score at or above this value shows a "Strong match" badge |