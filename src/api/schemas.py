from pydantic import BaseModel
from typing import List, Optional


class SessionStartResponse(BaseModel):
    session_id: str


class SessionEndRequest(BaseModel):
    session_id: str


class UploadPDFResponse(BaseModel):
    message: str
    filename: str
    total_pages: int
    total_chunks: int


class ChatRequest(BaseModel):
    session_id: str
    question: str


class SourceItem(BaseModel):
    page: Optional[int] = None
    source_file: Optional[str] = None
    chunk_id: str
    cosine_similarity: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]