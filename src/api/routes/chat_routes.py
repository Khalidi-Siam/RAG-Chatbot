from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.sql import func

from db.database import get_db
from db.models import Session
from pipeline.rag_pipeline import RAGPipeline


router = APIRouter()

rag = RAGPipeline()


# =========================
# REQUEST MODEL
# =========================
class ChatRequest(BaseModel):
    question: str


# =========================
# CHAT ENDPOINT
# =========================
@router.post("/{session_id}/chat")
def chat(
    session_id: str,
    payload: ChatRequest,
    db: DBSession = Depends(get_db)
):
    # 1. Validate session
    session_obj = db.query(Session).filter(Session.id == session_id).first()

    if not session_obj or not session_obj.is_active:
        raise HTTPException(
            status_code=404,
            detail="Session not found or inactive"
        )

    try:
        # 2. Update last activity (FIXED PROPERLY)
        session_obj.last_activity = func.now()
        db.commit()

        # 3. Run RAG pipeline
        result = rag.ask(
            session_id=session_id,
            question=payload.question
        )

        return {
            "session_id": session_id,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )