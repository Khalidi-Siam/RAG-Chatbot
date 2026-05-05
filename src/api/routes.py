import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from config.settings import settings
from pipeline.rag_pipeline import RAGPipeline
from api.schemas import (
    SessionStartResponse,
    SessionEndRequest,
    UploadPDFResponse,
    ChatRequest,
    ChatResponse
)

router = APIRouter()

# Singleton pipeline (shared for whole server runtime)
rag = RAGPipeline()


@router.post("/session/start", response_model=SessionStartResponse)
def start_session():
    session_id = rag.start_session()
    return {"session_id": session_id}


@router.post("/session/end")
def end_session(data: SessionEndRequest):
    try:
        rag.end_session(data.session_id)
        return {"message": "Session ended successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload/pdf", response_model=UploadPDFResponse)
def upload_pdf(file: UploadFile = File(...)):
    # Validate file extension
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    os.makedirs(settings.upload_dir, exist_ok=True)

    save_path = os.path.join(settings.upload_dir, file.filename)

    # Save PDF file
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ingest PDF into vector DB
    try:
        ingest_result = rag.ingest_pdf(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {str(e)}")

    return {
        "message": "PDF uploaded and indexed successfully.",
        "filename": file.filename,
        "total_pages": ingest_result["total_pages"],
        "total_chunks": ingest_result["total_chunks"]
    }


@router.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest):
    try:
        response = rag.ask(data.session_id, data.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))