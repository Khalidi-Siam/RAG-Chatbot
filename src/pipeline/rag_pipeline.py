from components.pdf_ingestor import PDFIngestor
from components.embedder import GeminiEmbedder
from components.faiss_store import FAISSVectorStore
from components.retriever import Retriever
from components.gemini_llm import GeminiLLM
from components.prompt_template import build_rag_prompt
from components.session_manager import SessionManager
from config.settings import settings
from logger import logging

import os


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        similarity_threshold: float = settings.similarity_threshold,
        top_k: int = settings.top_k
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        self.ingestor = PDFIngestor(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        self.embedder = GeminiEmbedder(model_name=settings.embedding_model)
        self.llm = GeminiLLM(model_name=settings.llm_model)

        self.session_manager = SessionManager()

    # =========================
    # SESSION MANAGEMENT
    # =========================

    def start_session(self) -> str:
        return self.session_manager.create_session()

    def end_session(self, session_id: str):
        self.session_manager.delete_session(session_id)

    # =========================
    # SESSION VECTOR STORE
    # =========================

    def _get_vectorstore(self, session_id: str):
        """
        Each session has its own FAISS DB folder.
        """
        faiss_path = os.path.join(
            settings.faiss_base_dir,
            f"session_{session_id}"
        )

        os.makedirs(faiss_path, exist_ok=True)

        return FAISSVectorStore(
            persist_path=faiss_path,
            collection_name="pdf_knowledge"
        )

    # =========================
    # INGESTION (SESSION BASED)
    # =========================

    def ingest_pdf(self, pdf_path: str, faiss_path: str):
        logging.info(f"Starting ingestion: {pdf_path}")

        ingestion_result = self.ingestor.ingest(pdf_path)
        chunks = ingestion_result["chunks"]

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_documents(texts)

        # ⚠️ IMPORTANT CHANGE: use session-specific vectorstore
        vectorstore = FAISSVectorStore(
            persist_path=faiss_path,
            collection_name="pdf_knowledge"
        )

        vectorstore.add_documents(chunks, embeddings)

        result = {
            "total_pages": len(ingestion_result["pages"]),
            "total_chunks": len(chunks)
        }

        logging.info(f"Ingestion complete: {result}")
        return result

    # =========================
    # CHAT (SESSION BASED)
    # =========================

    def _format_chat_history(self, history: list[dict]) -> str:
        lines = []
        for msg in history:
            lines.append(f"{msg['role'].upper()}: {msg['message']}")
        return "\n".join(lines)

    def ask(self, session_id: str, question: str) -> dict:
        logging.info(f"[Session {session_id[:8]}] Q: {question[:80]}")


        # store user message
        self.session_manager.add_message(session_id, "user", question)

        # =========================
        # SESSION-SPECIFIC VECTOR DB
        # =========================
        vectorstore = self._get_vectorstore(session_id)

        retriever = Retriever(
            vectorstore=vectorstore,
            similarity_threshold=self.similarity_threshold
        )

        query_embedding = self.embedder.embed_query(question)
        retrieved_chunks = retriever.retrieve(
            query_embedding,
            top_k=self.top_k
        )

        if not retrieved_chunks:
            answer = "Not found in the knowledge base."

            self.session_manager.add_message(session_id, "assistant", answer)

            return {
                "answer": answer,
                "sources": []
            }

        context_parts = []
        sources = []

        for chunk in retrieved_chunks:
            context_parts.append(f"(Page {chunk['page']}) {chunk['text']}")
            sources.append({
                "page": chunk["page"],
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                "cosine_similarity": chunk["cosine_similarity"]
            })

        context = "\n\n".join(context_parts)

        history = self.session_manager.get_history(session_id, last_n=6)
        chat_history_text = self._format_chat_history(history)

        prompt = build_rag_prompt(
            context=context,
            question=question,
            chat_history=chat_history_text
        )

        answer = self.llm.generate(prompt)

        self.session_manager.add_message(session_id, "assistant", answer)

        return {
            "answer": answer,
            "sources": sources
        }