from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

LOCAL_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

def get_response(
    messages,
    user_api_key=None
):

    api_key = LOCAL_API_KEY

    if not api_key:
        api_key = user_api_key

    if not api_key:
        return (
            "Please enter a valid Groq API Key."
        )

    client = Groq(
        api_key=api_key
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7
    )

    return response.choices[0].message.content