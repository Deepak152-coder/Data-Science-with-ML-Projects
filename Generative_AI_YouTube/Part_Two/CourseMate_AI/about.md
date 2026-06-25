# 📚 CourseMate AI

CourseMate AI is an AI-powered study assistant designed to help students learn more efficiently by interacting with their study materials through natural language. Instead of manually searching through lengthy documents, students can ask questions and receive accurate, context-aware answers in seconds.

---

## 🚀 Overview

Modern students rely on multiple sources of learning materials, including:

- 📄 Lecture Notes
- 📚 Textbooks
- 📑 PDFs
- 📝 Research Papers
- 📖 Study Guides

These resources are often lengthy and difficult to navigate. CourseMate AI simplifies the learning experience by allowing users to chat directly with their documents, making information retrieval fast, intuitive, and efficient.

---

## ✨ Features

- 🤖 AI-powered question answering
- 📄 Chat with PDF documents
- 📚 Supports multiple study materials
- 🔍 Semantic search using vector embeddings
- 💬 Context-aware responses
- ⚡ Fast document retrieval with Retrieval-Augmented Generation (RAG)
- 🎯 User-friendly interface
- 📖 Helps with revision and exam preparation

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python
- LangChain

### AI & LLM
- Mistral AI
- Hugging Face Embeddings

### Vector Database
- FAISS

### Document Processing
- PyPDF
- LangChain Document Loaders
- Recursive Character Text Splitter

### Environment Management
- Python Virtual Environment (uv / venv)
- python-dotenv

---

## 📂 Project Workflow

1. Upload one or more PDF documents.
2. Extract text from the documents.
3. Split the text into manageable chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in a FAISS vector database.
6. Retrieve the most relevant chunks based on the user's question.
7. Pass the retrieved context to the LLM.
8. Generate an accurate, context-aware answer.

---

## 🎯 Use Cases

- Study smarter from lecture notes
- Understand textbook concepts quickly
- Search research papers efficiently
- Prepare for exams
- Summarize lengthy documents
- Ask follow-up questions on study materials

---

## 📁 Project Structure

```
CourseMate-AI/
│
├── app.py
├── core.py
├── requirements.txt
├── .env
├── README.md
├── about.md
│
├── data/
│   └── PDFs
│
├── vectorstore/
│
├── utils/
│
└── assets/
```

---

## 📸 Preview

CourseMate AI provides a clean and interactive interface where students can upload study materials and chat with them using natural language.

---

## 🔮 Future Enhancements

- Multiple document collections
- Chat history
- Voice input
- PDF highlighting
- Citation support
- Image understanding
- Multi-modal document support
- Cloud deployment
- Authentication system
- Conversation memory

---

## 👨‍💻 Author

**Deepak Kumar**

Passionate about Artificial Intelligence, Machine Learning, Data Science, and building practical AI applications.

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub. Your support helps motivate future open-source projects.