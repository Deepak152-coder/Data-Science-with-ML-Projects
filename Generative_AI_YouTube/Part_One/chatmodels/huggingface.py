from dotenv import load_dotenv
import os

# cd "Part_One/chatmodels"
# python huggingface.py

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_KEY")
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is the capital of France?")
print(response.content)