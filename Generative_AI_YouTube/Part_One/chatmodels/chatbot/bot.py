from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
import os

load_dotenv()

# Personalities
personalities = {
    "Funny": "You are a funny AI assistant who loves jokes and humor.",
    "Angry": "You are an angry AI assistant. Respond aggressively but do not use abusive language.",
    "Teacher": "You are a patient teacher who explains concepts simply.",
    "Motivator": "You are a motivational coach who encourages users.",
    "Pirate": "You are a pirate. Speak like a pirate.",
    "Professional": "You are a professional AI assistant. Be concise and formal."
}

print("Choose Personality:")
for i, role in enumerate(personalities.keys(), start=1):
    print(f"{i}. {role}")

choice = int(input("Enter choice: "))

selected_role = list(personalities.keys())[choice - 1]

messages = [
    SystemMessage(content=personalities[selected_role])
]

model = ChatMistralAI(
    model_name="mistral-small-2506",
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.9
)

print(f"\n-------- Welcome to {selected_role} ChatBot --------")

while True:
    prompt = input("You: ")

    if prompt.lower() in ["0", "exit", "quit"]:
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("Bot:", response.content)