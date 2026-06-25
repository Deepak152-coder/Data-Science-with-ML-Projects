from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load Existing Chroma DB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    },
)

# LLM
llm = ChatMistralAI(
    model="mistral-small-2506"
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)

print("RAG System Created Successfully!")
print("Press 0 to exit.\n")

while True:
    query = input("You: ")

    if query == "0":
        break

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    # Create context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Prepare prompt
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    # Generate response
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")