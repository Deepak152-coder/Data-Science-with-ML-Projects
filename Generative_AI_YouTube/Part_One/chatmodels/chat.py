from dotenv import load_dotenv
load_dotenv()

# cd "Part_One/chatmodels"
# python chat.py

# Bahubali (It will run all the models)
# from langchain.chat_models import init_chat_model

# model = init_chat_model(
#     "llama-3.3-70b-versatile",
#     model_provider="groq"
# )

# response = model.invoke("Hello, how are you?")

# print(response.content)

# Mistral (It will run all the models)
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(
    model="mistral-small-latest",  
    temperature=0.1,
    max_tokens=20,
)

response = model.invoke("Write a poem on AI")
print(response.content)

# We can also run a specific model by using the model name and provider as follows:

# from langchain.chat_models import ChatModel
# from langchain.llama import LlamaChatModel
# from langchain.openai import OpenAIChatModel
# from langchain.palm import PalmChatModel
# from langchain.gemini import GeminiChatModel
# from langchain-mistralai import MistralChatModel  
# from langchain_groq import ChatGroq
# ......etc