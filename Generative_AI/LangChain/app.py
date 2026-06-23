from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

messages = []

print("\n===== SELECT PROVIDER =====")
print("1. Groq")
print("2. Mistral")

provider = input("\nEnter choice: ")

if provider == "1":

    print("\n===== GROQ MODELS =====")
    print("1. llama-3.3-70b-versatile")
    print("2. llama3-8b-8192")

    model_choice = input("\nSelect model: ")

    groq_models = {
        "1": "llama-3.3-70b-versatile",
        "2": "llama3-8b-8192"
    }

    llm = ChatGroq(
        model=groq_models.get(model_choice, "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY")
    )

elif provider == "2":

    print("\n===== MISTRAL MODELS =====")
    print("1. mistral-small-latest")
    print("2. mistral-large-latest")

    model_choice = input("\nSelect model: ")

    mistral_models = {
        "1": "mistral-small-latest",
        "2": "mistral-large-latest"
    }

    llm = ChatMistralAI(
        model=mistral_models.get(model_choice, "mistral-small-latest"),
        api_key=os.getenv("MISTRAL_API_KEY")
    )

else:
    print("Invalid Provider")
    exit()

print("\n===== CHAT STARTED =====")
print("Type 'exit' to quit")
print("Type 'clear' to clear memory")

while True:

    prompt = input("\nYou: ")

    if prompt.lower() == "exit":
        print("\nGoodbye!")
        break

    if prompt.lower() == "clear":
        messages.clear()
        print("\nMemory Cleared!")
        continue

    messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("AI:")
    print(response.content)

    print(f"\n[Memory Size: {len(messages)} messages]")