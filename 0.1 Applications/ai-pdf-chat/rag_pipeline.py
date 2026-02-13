import os
import shutil
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

from utils import vectorstore_dir_for

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

class RAGPipeline:
    def __init__(self, llm_temperature: float = 0.0):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
        self.llm = ChatOpenAI(temperature=llm_temperature, openai_api_key=OPENAI_KEY)
        self._vectorstores: Dict[str, FAISS] = {}  # cache vectorstores
        self.sessions: Dict[str, List] = {}        # session chat history

    def ingest_pdf(self, pdf_path: str, pdf_id: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        # load PDF and split
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)

        # create vectorstore
        vs = FAISS.from_documents(chunks, self.embeddings)

        # save index + PDF
        dirpath = vectorstore_dir_for(pdf_id)
        os.makedirs(dirpath, exist_ok=True)
        vs.save_local(dirpath)
        shutil.copy(pdf_path, os.path.join(dirpath, f"{pdf_id}.pdf"))

        self._vectorstores[pdf_id] = vs

    def _load_vectorstore(self, pdf_id: str) -> FAISS:
        if pdf_id in self._vectorstores:
            return self._vectorstores[pdf_id]

        dirpath = vectorstore_dir_for(pdf_id)
        if not os.path.exists(dirpath):
            raise ValueError(f"No vectorstore found for {pdf_id} at {dirpath}")

        # load FAISS index from disk safely
        vs = FAISS.load_local(dirpath, self.embeddings, allow_dangerous_deserialization=True)
        self._vectorstores[pdf_id] = vs
        return vs

    def get_retriever(self, pdf_id: str, k: int = 4):
        vs = self._load_vectorstore(pdf_id)
        return vs.as_retriever(search_kwargs={"k": k})

    def chat(self, pdf_id: str, query: str, session_id: Optional[str] = None, k: int = 4) -> Dict[str, Any]:
        retriever = self.get_retriever(pdf_id, k=k)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )

        history = self.sessions.get(session_id, []) if session_id else []
        result = qa_chain({"question": query, "chat_history": history})

        answer = result.get("answer") or result.get("result")
        source_docs: List[Document] = result.get("source_documents", [])

        if session_id:
            history.append((query, answer))
            self.sessions[session_id] = history

        sources = []
        for d in source_docs:
            meta = d.metadata or {}
            sources.append({
                "page": meta.get("page", meta.get("page_number")),
                "source_text": (d.page_content[:800] + "...") if len(d.page_content) > 800 else d.page_content
            })

        return {"answer": answer, "sources": sources, "raw": result}
