
# 📄 AI-powered PDF Reader with Chat (FastAPI + LangChain)

This project is an **AI-powered PDF reader** where users can upload PDFs and **interactively chat** with their content.  
It uses **LangChain**, **OpenAI LLMs**, and **FastAPI** to build a **Retrieval-Augmented Generation (RAG)** pipeline.  
The goal is to provide an industry-level learning project that demonstrates semantic search, context-aware answers, and conversational AI.

---

## 🚀 Features
- Upload a PDF and extract text
- Chunk documents and generate embeddings
- Store embeddings in **FAISS** (local vector database)
- Ask natural language questions about the PDF
- Semantic search + LLM to generate accurate answers
- Maintains **chat history** for contextual responses
- REST API built using **FastAPI**
- Optional **Streamlit UI** for interaction

---

## 🛠️ Tech Stack
- **Backend:** FastAPI
- **LLM Orchestration:** LangChain
- **Embeddings & Chat Models:** OpenAI (`text-embedding-3-small`, GPT-3.5/4)
- **Vector DB:** FAISS
- **PDF Processing:** LangChain’s PyPDFLoader
- **Frontend (optional):** Streamlit or React

---

## 📂 Project Structure
ai_pdf_chat/
│── main.py # FastAPI entrypoint
│── rag_pipeline.py # LangChain RAG pipeline
│── models.py # Pydantic request/response schemas
│── database.py # VectorDB/SQLite handling (optional)
│── streamlit_app.py # Optional frontend UI
│── requirements.txt # Dependencies
│── .env # API Keys


---

## ⚡ API Endpoints

| Endpoint               | Method | Description |
|------------------------|--------|-------------|
| `/upload_pdf`          | POST   | Upload a PDF, process text, generate embeddings |
| `/pdfs`                | GET    | List all uploaded PDFs |
| `/create_session`      | POST   | Create a chat session (for context-aware answers) |
| `/chat`                | POST   | Ask a question to a PDF, get context-aware answers |
| `/agent_chat`          | POST   | Agent-style query with multi-step reasoning over PDF chunks |

**Example `/chat` request body:**
```json
{
  "pdf_id": "pdf_6c5a3d41",
  "session_id": "sess_abc123xyz",
  "query": "Explain CNN in simple terms",
  "k": 3
}
```

⚡ How to Run Locally

Clone the repo:

```bash
git clone <your-repo-url>
cd ai_pdf_chat
```

Create virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Add .env with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

Run FastAPI server:

```bash
uvicorn main:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

Optional: run Streamlit frontend:

```bash
streamlit run streamlit_app.py
```

📖 Notes

- Use session_id for follow-up questions to maintain context.
- k controls how many chunks to retrieve from the PDF (default 4).
- FAISS is used for fast, local semantic search.

🔗 References

- [LangChain Docs](https://www.langchain.com/docs/)
- [OpenAI API Docs](https://platform.openai.com/docs/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
