from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

while True:
    user_input = input("Enter your prompt: ")

    response = model.invoke(user_input)

    print("Response:", response.content)