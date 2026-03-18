import re
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "from", "this", "that", "these", "those", "is", "are", "was", "were", "be", "been",
    "it", "its", "we", "you", "your", "our", "they", "their", "will", "can", "may", "should",
    "must", "not", "but", "if", "than", "then", "also", "such", "using", "use", "used",
    "experience", "skills", "required", "responsibilities", "preferred", "plus",
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+\-#/.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, min_word_len: int) -> List[str]:
    tokens = []
    for w in normalize(text).split():
        w_clean = w.strip(".")
        if len(w_clean) < min_word_len:
            continue
        if w_clean in STOPWORDS:
            continue
        tokens.append(w_clean)
    return tokens


def keyword_counts(text: str, min_word_len: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in tokenize(text, min_word_len=min_word_len):
        counts[t] = counts.get(t, 0) + 1
    return counts


def top_keywords(counts: Dict[str, int], top_k: int) -> List[Tuple[str, int]]:
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_k]


def tfidf_similarity(resume_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([resume_text, jd_text])
    sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return float(sim)


def analyze_match(
    resume_text: str,
    jd_text: str,
    top_k: int = 20,
    min_word_len: int = 4,
) -> dict:
    resume_norm = normalize(resume_text)
    jd_norm = normalize(jd_text)

    sim = tfidf_similarity(resume_norm, jd_norm)

    jd_counts = keyword_counts(jd_norm, min_word_len=min_word_len)
    resume_counts = keyword_counts(resume_norm, min_word_len=min_word_len)

    jd_top = top_keywords(jd_counts, top_k=top_k)

    top_jd_df = pd.DataFrame(jd_top, columns=["keyword", "count"])

    missing = [(kw, cnt) for kw, cnt in jd_top if kw not in resume_counts]
    missing_df = pd.DataFrame(missing, columns=["keyword", "count"])

    if len(jd_top) == 0:
        coverage = 0.0
    else:
        present = sum(1 for kw, _ in jd_top if kw in resume_counts)
        coverage = (present / len(jd_top)) * 100.0

    # Scoring formula: similarity contributes up to 60, coverage up to 40 → range 0–100
    match_score = (sim * 60.0) + (coverage * 0.40)

    suggestions = _build_suggestions(
        missing_keywords=[kw for kw, _ in missing],
        coverage=coverage,
    )

    return {
        "similarity": sim,
        "keyword_coverage": coverage,
        "match_score": max(0.0, min(100.0, match_score)),
        "top_jd_keywords": top_jd_df,
        "missing_keywords": missing_df,
        "suggestions": suggestions,
    }


def _build_suggestions(missing_keywords: List[str], coverage: float) -> str:
    if coverage >= 80:
        return (
            "Great coverage! Double-check that your resume includes specific, measurable achievements "
            "(numbers, impact) that match the job responsibilities."
        )
    if not missing_keywords:
        return (
            "Your resume already contains most of the top keywords. Improve clarity by adding impact "
            "metrics (e.g., reduced load time by 30%)."
        )
    sample = ", ".join(missing_keywords[:8])
    return (
        f"Consider adding relevant keywords you truly have experience with. "
        f"Missing examples: {sample}. "
        "If you don't have them yet, consider learning them or emphasizing equivalent skills/projects."
    )
