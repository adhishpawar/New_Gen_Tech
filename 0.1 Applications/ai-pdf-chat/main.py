from pydantic import BaseModel
import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()

from utils import generate_id, save_upload_fileobj
from db import add_pdf, list_pdfs, get_pdf
from rag_pipeline import RAGPipeline
from agent import SimpleAgent

app = FastAPI(title="AI PDF Chat (FastAPI + LangChain)")

# initialize pipeline
pipeline = RAGPipeline()
agent = SimpleAgent(llm=pipeline.llm)

class ChatRequest(BaseModel):
    pdf_id: str
    query: str
    session_id: str | None = None
    k: int = 4

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # save file
    filename = file.filename
    pdf_id = generate_id()
    saved_path = save_upload_fileobj(file.file, f"{pdf_id}_{filename}")
    # ingest
    pipeline.ingest_pdf(saved_path, pdf_id)
    # track metadata
    add_pdf(pdf_id, filename, saved_path)
    return {"message": "uploaded and processed", "pdf_id": pdf_id}

@app.get("/pdfs")
def pdfs():
    return {"pdfs": list_pdfs()}

@app.post("/chat")
async def chat(request: ChatRequest):
    # validate pdf exists
    meta = get_pdf(request.pdf_id)
    if not meta:
        return JSONResponse({"error": "pdf_id not found"}, status_code=404)

    res = pipeline.chat(
        pdf_id=request.pdf_id,
        query=request.query,
        session_id=request.session_id,
        k=request.k
    )
    return res

@app.post("/create_session")
def create_session():
    sid = generate_id(prefix="sess")
    pipeline.sessions[sid] = []
    return {"session_id": sid}

@app.post("/agent_chat")
async def agent_chat(pdf_id: str = Form(...), query: str = Form(...), k: int = Form(4)):
    meta = get_pdf(pdf_id)
    if not meta:
        return JSONResponse({"error": "pdf_id not found"}, status_code=404)
    retriever = pipeline.get_retriever(pdf_id, k=k)
    # agent returns a raw string answer (from SimpleAgent.run)
    answer = agent.run(retriever, query, k=k)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
