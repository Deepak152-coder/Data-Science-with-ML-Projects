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
.main {
    padding-top: 1rem;
}
.block-container {
    padding-top: 1rem;
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

# Clean titles
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
    "Number of Recommendations",
    min_value=1,
    max_value=20,
    value=10
)

st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ About Project"):

    st.write("""
    This project uses:

    - Natural Language Processing
    - TF-IDF Vectorization
    - Cosine Similarity
    - Content-Based Filtering
    - Streamlit Deployment

    Built for Movie Recommendation.
    """)

# =====================================
# RECOMMENDATION FUNCTION
# =====================================

def recommend(title, n=10):

    try:

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
                "Similarity Score (%)": round(
                    sim_scores[i] * 100,
                    2
                )
            })

        return recommendations

    except Exception:
        return None

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

# =====================================
# MOVIE INFO
# =====================================

if selected_movie:

    st.info(f"Selected Movie: {selected_movie}")

# =====================================
# RECOMMEND BUTTON
# =====================================

if st.button("🎯 Recommend Movies"):

    recommendations = recommend(
        selected_movie,
        num_recommendations
    )

    if recommendations is None:

        st.error("Movie not found!")

    else:

        st.subheader("🎥 Recommended Movies")

        rec_df = pd.DataFrame(
            recommendations
        )

        st.dataframe(
            rec_df,
            use_container_width=True
        )

        st.subheader("⭐ Similarity Scores")

        for _, row in rec_df.iterrows():

            st.write(
                f"**{row['Movie']}**"
            )

            score = min(
                int(row["Similarity Score (%)"]),
                100
            )

            st.progress(score)

            st.caption(
                f"Similarity Score: {row['Similarity Score (%)']}%"
            )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown("""
### 🚀 Technologies Used

- Python
- Pandas
- Scikit-Learn
- TF-IDF Vectorizer
- Cosine Similarity
- Streamlit
- NLP

Made with ❤️ by Deepak Kumar
""")