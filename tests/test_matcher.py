import pytest

from matcher import (
    analyze_match,
    keyword_counts,
    normalize,
    tfidf_similarity,
    tokenize,
    top_keywords,
)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def test_normalize_lowercases():
    assert normalize("Python SQL") == "python sql"


def test_normalize_removes_punctuation():
    result = normalize("Hello, World! (test)")
    assert "," not in result
    assert "!" not in result
    assert "(" not in result


def test_normalize_collapses_whitespace():
    assert normalize("  hello   world  ") == "hello world"


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def test_tokenize_filters_short_words():
    tokens = tokenize("I am a Python developer", min_word_len=4)
    assert "i" not in tokens
    assert "am" not in tokens
    assert "python" in tokens


def test_tokenize_filters_stopwords():
    tokens = tokenize("the experience with skills", min_word_len=3)
    assert "the" not in tokens
    assert "experience" not in tokens


# ---------------------------------------------------------------------------
# keyword_counts
# ---------------------------------------------------------------------------

def test_keyword_counts_counts_correctly():
    counts = keyword_counts("python python sql", min_word_len=3)
    assert counts["python"] == 2
    assert counts["sql"] == 1


# ---------------------------------------------------------------------------
# top_keywords
# ---------------------------------------------------------------------------

def test_top_keywords_returns_correct_count():
    counts = {"python": 3, "sql": 2, "react": 1, "docker": 4}
    result = top_keywords(counts, top_k=2)
    assert len(result) == 2
    assert result[0][0] == "docker"


# ---------------------------------------------------------------------------
# tfidf_similarity
# ---------------------------------------------------------------------------

def test_tfidf_similarity_identical_texts():
    sim = tfidf_similarity("python developer sql", "python developer sql")
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_tfidf_similarity_completely_different():
    sim = tfidf_similarity("python sql react", "cooking baking pastry")
    assert sim == pytest.approx(0.0, abs=1e-6)


def test_tfidf_similarity_partial_overlap():
    sim = tfidf_similarity("python sql react", "python java databases")
    assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# analyze_match
# ---------------------------------------------------------------------------

def test_analyze_match_returns_expected_keys():
    resume = "Python SQL React built dashboards collaborated teams"
    jd = "Looking for a Python developer with SQL and APIs React is a plus"
    result = analyze_match(resume, jd, top_k=10, min_word_len=3)

    assert "match_score" in result
    assert "similarity" in result
    assert "keyword_coverage" in result
    assert "top_jd_keywords" in result
    assert "missing_keywords" in result
    assert "suggestions" in result


def test_analyze_match_score_in_range():
    resume = "Python SQL React built dashboards"
    jd = "Python developer SQL APIs React"
    result = analyze_match(resume, jd, top_k=10, min_word_len=3)
    assert 0.0 <= result["match_score"] <= 100.0


def test_analyze_match_perfect_overlap():
    text = "python sql react docker kubernetes aws"
    result = analyze_match(text, text, top_k=10, min_word_len=3)
    assert result["keyword_coverage"] == pytest.approx(100.0)
    assert result["match_score"] == pytest.approx(100.0, abs=1.0)


def test_analyze_match_no_overlap():
    resume = "cooking baking pastry desserts"
    jd = "python sql react docker"
    result = analyze_match(resume, jd, top_k=10, min_word_len=3)
    assert result["keyword_coverage"] == pytest.approx(0.0)
    assert result["similarity"] == pytest.approx(0.0, abs=1e-6)


def test_analyze_match_missing_keywords_subset_of_top_jd():
    resume = "python sql"
    jd = "python sql react docker kubernetes"
    result = analyze_match(resume, jd, top_k=5, min_word_len=3)
    top_kws = set(result["top_jd_keywords"]["keyword"])
    missing_kws = set(result["missing_keywords"]["keyword"])
    assert missing_kws.issubset(top_kws)


def test_analyze_match_dataframe_columns():
    resume = "python sql"
    jd = "python developer with sql and react"
    result = analyze_match(resume, jd, top_k=5, min_word_len=3)
    assert list(result["top_jd_keywords"].columns) == ["keyword", "count"]
    assert list(result["missing_keywords"].columns) == ["keyword", "count"]


def test_analyze_match_suggestions_is_string():
    resume = "python sql"
    jd = "python developer with sql"
    result = analyze_match(resume, jd, top_k=5, min_word_len=3)
    assert isinstance(result["suggestions"], str)
    assert len(result["suggestions"]) > 0
