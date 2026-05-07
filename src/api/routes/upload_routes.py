import os
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session as DBSession

from db.database import get_db
from db.models import Session, KnowledgeBase, Document
from config.settings import settings
from services.lock_manager import lock_manager

from pipeline.rag_pipeline import RAGPipeline


router = APIRouter()

rag = RAGPipeline()  # shared pipeline object


@router.post("/{session_id}/upload")
def upload_pdf(
    session_id: str,
    file: UploadFile = File(...),
    db: DBSession = Depends(get_db)
):
    # 1. Validate session
    session_obj = db.query(Session).filter(Session.id == session_id).first()
    if not session_obj or not session_obj.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive.")

    # 2. Validate PDF file
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # 3. Get KB info
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.session_id == session_obj.id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found for this session.")

    # 4. Save PDF inside session upload folder
    os.makedirs(kb.upload_path, exist_ok=True)

    stored_filename = file.filename
    save_path = os.path.join(kb.upload_path, stored_filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 5. Insert document row in DB (status = processing)
    file_size = os.path.getsize(save_path)
    doc = Document(
        knowledge_base_id=kb.id,
        original_filename=file.filename,
        stored_filename=stored_filename,
        file_path=save_path,
        file_size=file_size,
        status="processing"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # 6. Acquire session lock to avoid FAISS conflict
    session_lock = lock_manager.get_lock(session_id)

    with session_lock:
        try:
            # Ingest into FAISS session folder
            ingest_result = rag.ingest_pdf(
                pdf_path=save_path,
                faiss_path=kb.faiss_path
            )

            doc.total_pages = ingest_result["total_pages"]
            doc.total_chunks = ingest_result["total_chunks"]
            doc.status = "indexed"

            db.commit()

        except Exception as e:
            doc.status = "failed"
            db.commit()
            raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {str(e)}")

    return {
        "message": "PDF uploaded and indexed successfully.",
        "document_id": str(doc.id),
        "filename": file.filename,
        "file_size": file_size,
        "total_pages": doc.total_pages,
        "total_chunks": doc.total_chunks,
        "status": doc.status
    }