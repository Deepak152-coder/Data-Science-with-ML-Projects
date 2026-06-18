import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import linear_kernel

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>
.movie-card {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# FILE PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

# =====================================
# LOAD FILES
# =====================================

@st.cache_resource
def load_files():

    with open(BASE_DIR / "df.pkl", "rb") as f:
        df = pickle.load(f)

    with open(BASE_DIR / "indices.pkl", "rb") as f:
        indices = pickle.load(f)

    with open(BASE_DIR / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    return df, indices, tfidf_matrix


df, indices, tfidf_matrix = load_files()

df["title"] = df["title"].fillna("").astype(str)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📊 Dashboard")

st.sidebar.metric(
    "Total Movies",
    len(df)
)

st.sidebar.metric(
    "Unique Movies",
    df["title"].nunique()
)

num_recommendations = st.sidebar.slider(
    "🎯 Number of Recommendations",
    min_value=1,
    max_value=20,
    value=10
)

st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ About Project"):
    st.write("""
    This project uses:

    • NLP

    • TF-IDF Vectorization

    • Cosine Similarity

    • Content-Based Recommendation

    • Streamlit Deployment
    """)

# =====================================
# RECOMMENDATION FUNCTION
# =====================================

def recommend(title, n=10):

    idx = indices[title]

    sim_scores = linear_kernel(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    movie_indices = sim_scores.argsort()[::-1][1:n+1]

    recommendations = []

    for i in movie_indices:

        recommendations.append({
            "Movie": df.iloc[i]["title"],
            "Score": round(sim_scores[i] * 100, 2)
        })

    return recommendations

# =====================================
# MAIN UI
# =====================================

st.title("🎬 AI Movie Recommendation System")

st.markdown("""
Discover movies similar to your favorites using
**Natural Language Processing (TF-IDF)** and
**Cosine Similarity**.
""")

movie_list = sorted(
    df["title"]
    .dropna()
    .astype(str)
    .unique()
)

selected_movie = st.selectbox(
    "🔍 Search and Select a Movie",
    movie_list
)

st.info(f"Selected Movie: {selected_movie}")

# =====================================
# RECOMMEND BUTTON
# =====================================

if st.button("🎯 Recommend Movies"):

    st.balloons()

    recommendations = recommend(
        selected_movie,
        num_recommendations
    )

    st.subheader("🎥 Recommended Movies")

    rec_df = pd.DataFrame(recommendations)

    st.dataframe(
        rec_df,
        use_container_width=True
    )

    st.metric(
        "Recommendations Generated",
        len(rec_df)
    )

    st.subheader("⭐ Recommendation Strength")

    for rec in recommendations:

        movie = rec["Movie"]
        score = rec["Score"]

        st.markdown(
            f"""
            <div class="movie-card">
                <h4>🎬 {movie}</h4>
                <p>Similarity Score: {score}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(min(int(score), 100))

        if score >= 80:
            st.success("🔥 Excellent Match")

        elif score >= 60:
            st.warning("⭐ Good Match")

        elif score >= 40:
            st.info("👍 Moderate Match")

        else:
            st.caption("🎲 Low Match")

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown("""
### 🚀 Technologies Used

- Python
- Pandas
- Scikit-Learn
- NLP
- TF-IDF Vectorizer
- Cosine Similarity
- Streamlit

Made with ❤️ by Deepak Kumar
""")