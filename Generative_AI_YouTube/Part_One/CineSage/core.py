from dotenv import load_dotenv
load_dotenv()

import os
from typing import List, Optional

from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)

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

movie_description = input("Enter movie description: ")

final_prompt = prompt.invoke({
    "movie_description": movie_description,
    "format_instructions": parser.get_format_instructions()
})

model = ChatMistralAI(
    model_name="mistral-small-2506",
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.7
)

response = model.invoke(final_prompt)

parsed_response = parser.parse(response.content)

print("\n🎬 Extracted Movie Information\n")
print(parsed_response)

# Optional: Print as dictionary
# print(parsed_response.model_dump())