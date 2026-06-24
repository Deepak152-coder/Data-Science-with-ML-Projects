import streamlit as st
from dotenv import load_dotenv
import os

from pydantic import BaseModel
from typing import List, Optional

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="CineSage",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: #ff4b4b;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<p class="title">🎬 CineSage</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-Powered Movie Information Extractor</p>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Pydantic Schema
# --------------------------------------------------
class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)

# --------------------------------------------------
# Model
# --------------------------------------------------
model = ChatMistralAI(
    model_name="mistral-small-2506",
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.7
)

# --------------------------------------------------
# Prompt
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert movie information extraction assistant.

Analyze the movie description provided by the user and extract:

- title
- release_year
- genre
- director
- cast
- rating
- summary

Rules:
1. Return ONLY valid JSON.
2. Follow the exact schema provided below.
3. If a field is missing, use null for single values and [] for lists.
4. Keep the summary concise (2-3 sentences maximum).
5. Do not add explanations, markdown, or extra text.

{format_instructions}
"""
    ),
    (
        "human",
        "{movie_description}"
    )
])

# --------------------------------------------------
# Input
# --------------------------------------------------
movie_description = st.text_area(
    "🎥 Enter Movie Description",
    height=250,
    placeholder="Paste a movie description here..."
)

# --------------------------------------------------
# Analyze Button
# --------------------------------------------------
if st.button("🚀 Analyze Movie", use_container_width=True):

    if movie_description.strip():

        try:
            with st.spinner("Analyzing movie..."):

                final_prompt = prompt.invoke({
                    "movie_description": movie_description,
                    "format_instructions": parser.get_format_instructions()
                })

                response = model.invoke(final_prompt)

                movie = parser.parse(response.content)

            st.success("Movie analyzed successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("🎬 Title", movie.title)

                st.metric(
                    "📅 Release Year",
                    movie.release_year if movie.release_year else "N/A"
                )

            with col2:
                st.metric(
                    "⭐ IMDb Rating",
                    movie.rating if movie.rating else "N/A"
                )

                st.write("**🎭 Genre**")
                st.write(", ".join(movie.genre) if movie.genre else "N/A")

            st.write("**🎬 Director**")
            st.write(movie.director if movie.director else "N/A")

            st.write("**👥 Cast**")
            st.write(", ".join(movie.cast) if movie.cast else "N/A")

            st.write("**📝 Summary**")
            st.info(movie.summary)

            with st.expander("📦 Raw Parsed Output"):
                st.json(movie.model_dump())

        except Exception as e:
            st.error(f"Error while parsing response: {e}")

    else:
        st.warning("Please enter a movie description.")