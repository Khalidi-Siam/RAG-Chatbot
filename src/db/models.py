import uuid
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from db.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_activity = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    ended_at = Column(DateTime(timezone=True), nullable=True)

    knowledge_base = relationship(
        "KnowledgeBase",
        back_populates="session",
        uselist=False,
        cascade="all, delete"
    )


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), unique=True, nullable=False)

    faiss_path = Column(Text, nullable=False)
    upload_path = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session = relationship("Session", back_populates="knowledge_base")

    documents = relationship(
        "Document",
        back_populates="knowledge_base",
        cascade="all, delete"
    )


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    knowledge_base_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)

    original_filename = Column(Text, nullable=False)
    stored_filename = Column(Text, nullable=False)
    file_path = Column(Text, nullable=False)

    total_pages = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=True)

    status = Column(String(30), nullable=False, default="indexed")

    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")