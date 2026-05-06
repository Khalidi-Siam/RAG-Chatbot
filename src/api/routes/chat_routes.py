from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession
from datetime import datetime, timezone

from db.database import get_db
from db.models import Session, KnowledgeBase
from pipeline.rag_pipeline import RAGPipeline
from components.session_manager import SessionManager
from config.settings import settings
import os


router = APIRouter()

rag = RAGPipeline()
session_manager = SessionManager()


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
        session_obj.last_activity = datetime.now(timezone.utc)
        db.commit()
        
        # 3. Get FAISS path
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.session_id == session_id).first()
        faiss_path = kb.faiss_path if kb else os.path.join(settings.faiss_base_dir, f"session_{session_id}")

        # 4. Get chat history from memory
        history = session_manager.get_history(session_id, last_n=6)

        # 5. Run RAG pipeline (pure stateless)
        result = rag.ask(
            question=payload.question,
            faiss_path=faiss_path,
            chat_history=history
        )

        # 6. Update message memory
        session_manager.add_message(session_id, "user", payload.question)
        session_manager.add_message(session_id, "assistant", result["answer"])

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