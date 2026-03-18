import streamlit as st

from matcher import analyze_match

st.set_page_config(page_title="AI Resume ↔ Job Matcher", layout="wide")

st.title("AI Resume ↔ Job Matcher")
st.write(
    "Paste your resume and a job description. "
    "The app will score the match, highlight keywords, and suggest improvements."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Resume")
    resume_text = st.text_area(
        "Paste resume text here",
        height=320,
        placeholder="Example: Python, SQL, React, built dashboards, collaborated with teams...",
    )

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste job description here",
        height=320,
        placeholder="Example: Looking for a Python developer with SQL, APIs, AWS, and teamwork skills...",
    )

st.divider()

s1, s2, s3 = st.columns(3)
with s1:
    top_k = st.slider("Number of keywords to show", min_value=5, max_value=50, value=20, step=5)
with s2:
    min_word_len = st.slider("Minimum keyword length", min_value=3, max_value=10, value=4, step=1)
with s3:
    threshold = st.slider(
        "Strong-match threshold (0–100)",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        help="Scores at or above this value are shown as a strong match.",
    )

if st.button("Analyze Match", type="primary", use_container_width=True):
    if not resume_text.strip() or not jd_text.strip():
        st.error("Please paste both your resume and the job description before analyzing.")
    else:
        result = analyze_match(
            resume_text=resume_text,
            jd_text=jd_text,
            top_k=top_k,
            min_word_len=min_word_len,
        )

        st.subheader("Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Match Score (0–100)", f"{result['match_score']:.1f}")
        c2.metric("TF-IDF Cosine Similarity", f"{result['similarity']:.3f}")
        c3.metric("JD Keyword Coverage", f"{result['keyword_coverage']:.1f}%")

        st.caption(
            "Tip: Aim for higher keyword coverage. "
            "Only add skills you genuinely have—never fabricate experience."
        )

        kc1, kc2 = st.columns(2)
        with kc1:
            st.markdown("### Top JD Keywords")
            st.dataframe(result["top_jd_keywords"], use_container_width=True, hide_index=True)
        with kc2:
            st.markdown("### Missing Keywords")
            st.dataframe(result["missing_keywords"], use_container_width=True, hide_index=True)

        st.markdown("### Suggestions")
        st.write(result["suggestions"])

        if result["match_score"] >= threshold:
            st.success(
                f"Strong match! Score {result['match_score']:.1f} meets your threshold of {threshold}."
            )
        else:
            st.warning(
                f"Weaker match. Score {result['match_score']:.1f} is below your threshold of {threshold}. "
                "Consider tailoring your resume further."
            )
