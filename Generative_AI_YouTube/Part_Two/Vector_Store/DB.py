from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"topic": "Python"},
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"topic": "Pandas"},
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"topic": "Deep Learning"},
    ),
]

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db",
)

result = vector_store.similarity_search("What is used for data analysis?",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retrivere = vector_store.as_retriever()

docs = retrivere.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)