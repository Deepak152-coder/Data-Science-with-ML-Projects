import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import linear_kernel

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# ==========================
# Load Data
# ==========================
@st.cache_resource
def load_files():

    with open("df.pkl", "rb") as f:
        df = pickle.load(f)

    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)

    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    return df, indices, tfidf_matrix


df, indices, tfidf_matrix = load_files()

# Clean title column
df["title"] = df["title"].fillna("").astype(str)

# ==========================
# Recommendation Function
# ==========================
def recommend(title, n=10):

    try:
        idx = indices[title]

        sim_scores = linear_kernel(
            tfidf_matrix[idx],
            tfidf_matrix
        ).flatten()

        movie_indices = sim_scores.argsort()[::-1][1:n+1]

        return df["title"].iloc[movie_indices].tolist()

    except Exception:
        return ["Movie not found"]


# ==========================
# UI
# ==========================
st.title("🎬 Movie Recommendation System")
st.markdown(
    "Get movie recommendations based on movie content similarity using **TF-IDF**."
)

movie_list = sorted(df["title"].dropna().astype(str).unique())

selected_movie = st.selectbox(
    "Select a Movie",
    movie_list
)

num_recommendations = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=20,
    value=10
)

if st.button("Recommend Movies"):

    recommendations = recommend(
        selected_movie,
        num_recommendations
    )

    st.subheader("Recommended Movies")

    if recommendations[0] == "Movie not found":
        st.error("Movie not found in database.")
    else:
        for i, movie in enumerate(recommendations, 1):
            st.write(f"**{i}.** {movie}")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("Built with Streamlit | NLP Movie Recommendation System")