import os
import shutil
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from db.database import get_db
from db.models import Session, KnowledgeBase
from config.settings import settings
from components import session_manager


router = APIRouter()


@router.post("/start")
def start_session(db: DBSession = Depends(get_db)):
    """
    Creates a new anonymous session + creates its KB folder paths.
    """

    # 1. Create session row
    new_session = Session(is_active=True)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    session_id = str(new_session.id)

    # 2. Create folder paths
    upload_path = os.path.join(settings.upload_dir, f"session_{session_id}")
    faiss_path = os.path.join(settings.faiss_base_dir, f"session_{session_id}")

    os.makedirs(upload_path, exist_ok=True)
    os.makedirs(faiss_path, exist_ok=True)

    # 3. Create KB row
    kb = KnowledgeBase(
        session_id=new_session.id,
        upload_path=upload_path,
        faiss_path=faiss_path
    )

    db.add(kb)
    db.commit()

    return {
        "message": "Session created successfully",
        "session_id": session_id,
        "upload_path": upload_path,
        "faiss_path": faiss_path
    }


@router.post("/{session_id}/end")
def end_session(session_id: str, db: DBSession = Depends(get_db)):
    """
    Ends session and deletes all files + faiss db related to it.
    """

    session_obj = db.query(Session).filter(Session.id == session_id).first()

    if not session_obj:
        raise HTTPException(status_code=404, detail="Session not found.")

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.session_id == session_obj.id).first()

    if kb:
        # delete upload directory
        if os.path.exists(kb.upload_path):
            shutil.rmtree(kb.upload_path)

        # delete faiss directory
        if os.path.exists(kb.faiss_path):
            shutil.rmtree(kb.faiss_path)

    # delete session (cascade deletes kb + docs)
    db.delete(session_obj)
    db.commit()

    # clear memory
    session_manager.delete_session_memory(session_id)

    return {"message": "Session ended and cleaned up successfully."}

@router.get("/{session_id}/messages")
def get_session_messages(session_id: str):
    """
    Retrieve the chat history/messages for a given session at any point.
    """
    # Fetch up to the last 50 messages, or you can adjust `last_n` as needed
    history = session_manager.get_history(session_id, last_n=50)
    
    if not history:
        return {"session_id": session_id, "messages": [], "message": "No messages found for this session."}
        
    return {
        "session_id": session_id,
        "messages": history
    }
