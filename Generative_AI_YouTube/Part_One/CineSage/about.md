# 🎬 CineSage

## About the Project

CineSage is an AI-powered movie information extraction system designed to transform unstructured movie descriptions into structured, database-ready records.

The application takes a raw paragraph describing a movie, analyzes the content using Large Language Models (LLMs), extracts key information, generates a concise summary, and returns the results in JSON format for seamless storage and integration.

---

## Features

* Accepts raw movie descriptions as input
* Extracts important movie metadata

  * Title
  * Genre
  * Director
  * Cast
  * Release Year
  * Language
  * Runtime
  * IMDb Rating
  * Plot Elements
* Generates a clean and concise movie summary
* Produces structured JSON output
* Ready for database storage and downstream applications

---

## Example Workflow

### Input Movies

### 🎥 Movie 1: Interstellar

**Paragraph:**

Interstellar is a visually stunning science fiction epic directed by Christopher Nolan. Released in 2014, the film stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, and Michael Caine. The story revolves around a group of astronauts who travel through a wormhole near Saturn in search of a new home for humanity as Earth faces environmental collapse. The movie was widely appreciated for its emotional depth, scientific accuracy, and Hans Zimmer's powerful soundtrack. It holds a rating of 8.6 on IMDb and is often considered one of the greatest sci-fi films of the 21st century.

---

### 🎥 Movie 2: The Dark Knight

**Paragraph:**

The Dark Knight is a critically acclaimed superhero film directed by Christopher Nolan and released in 2008. Starring Christian Bale, Heath Ledger, Aaron Eckhart, and Gary Oldman, the movie follows Batman as he battles the chaotic criminal mastermind known as the Joker in Gotham City. Heath Ledger's unforgettable performance earned him a posthumous Academy Award for Best Supporting Actor. Praised for its storytelling, action sequences, and psychological depth, the film holds an IMDb rating of 9.0 and is regarded as one of the greatest superhero movies ever made.

---

### 🎥 Movie 3: The Shawshank Redemption

**Paragraph:**

The Shawshank Redemption is a drama film directed by Frank Darabont and released in 1994. The movie stars Tim Robbins and Morgan Freeman and tells the inspiring story of Andy Dufresne, a banker who is wrongly convicted of murder and sentenced to life imprisonment. Through patience, intelligence, and hope, Andy forms meaningful friendships and ultimately seeks freedom. The film received widespread critical acclaim for its emotional storytelling, memorable performances, and powerful themes of hope and perseverance. It currently holds an IMDb rating of 9.3, making it one of the highest-rated films of all time.

---

## Processing Pipeline

1. Accept raw movie description text
2. Analyze content using an LLM
3. Extract structured movie metadata
4. Generate a concise movie summary
5. Convert extracted information into JSON format
6. Store the structured output in a database

---

## Example Output

```json
{
  "title": "Interstellar",
  "genre": "Science Fiction",
  "director": "Christopher Nolan",
  "cast": [
    "Matthew McConaughey",
    "Anne Hathaway",
    "Jessica Chastain",
    "Michael Caine"
  ],
  "release_year": 2014,
  "imdb_rating": 8.6,
  "summary": "A team of astronauts travels through a wormhole in search of a new home for humanity as Earth faces environmental collapse."
}
```

---

## Tech Stack

* Python
* LangChain
* LangGraph
* Pydantic
* JSON Parsing
* Mistral AI
* OpenAI
* Google Gemini
* Groq
* Hugging Face
* FastAPI
* Streamlit
* SQL / NoSQL Databases

---

## Use Cases

* Movie Catalog Management
* Entertainment Recommendation Systems
* Content Management Platforms
* Film Analytics Dashboards
* Automated Metadata Generation
* Movie Search and Discovery Systems

---

## Goal

To automate the process of converting unstructured movie descriptions into structured, searchable, and database-ready information using Artificial Intelligence.
