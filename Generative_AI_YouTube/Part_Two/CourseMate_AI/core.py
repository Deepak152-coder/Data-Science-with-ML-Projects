from dotenv import load_dotenv
import os

from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loader = PyPDFLoader("Deep_Learning.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)  

chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI that summarizes the provided text."),
        ("human", "{data}")
    ]
)

model = ChatMistralAI(
    model="mistral-small-2506",
    api_key=os.getenv("MISTRAL_API_KEY")
)

prompt = template.format_messages(data= docs)

result = model.invoke(prompt)

print(result.content)