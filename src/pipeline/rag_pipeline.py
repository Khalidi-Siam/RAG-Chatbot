from components.pdf_ingestor import PDFIngestor
from components.embedder import GeminiEmbedder
from components.faiss_store import FAISSVectorStore
from components.retriever import Retriever
from components.gemini_llm import GeminiLLM
from components.prompt_template import build_rag_prompt
from components.session_manager import SessionManager


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        similarity_threshold: float = 0.6,
        top_k: int = 5
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        self.ingestor = PDFIngestor(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.embedder = GeminiEmbedder(model_name="gemini-embedding-001")

        self.vectorstore = FAISSVectorStore(
            persist_path="faiss_db",
            collection_name="pdf_knowledge"
        )

        self.retriever = Retriever(
            vectorstore=self.vectorstore,
            similarity_threshold=self.similarity_threshold
        )

        self.llm = GeminiLLM(model_name="gemini-2.5-flash")

        # NEW: session manager
        self.session_manager = SessionManager()

    def start_session(self) -> str:
        return self.session_manager.create_session()

    def end_session(self, session_id: str):
        self.session_manager.delete_session(session_id)

    def ingest_pdf(self, pdf_path: str):
        ingestion_result = self.ingestor.ingest(pdf_path)
        chunks = ingestion_result["chunks"]

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_documents(texts)

        self.vectorstore.add_documents(chunks, embeddings)

        return {
            "total_pages": len(ingestion_result["pages"]),
            "total_chunks": len(chunks)
        }

    def _format_chat_history(self, history: list[dict]) -> str:
        """
        Convert session history list into readable format for prompt.
        """
        lines = []
        for msg in history:
            role = msg["role"].upper()
            lines.append(f"{role}: {msg['message']}")
        return "\n".join(lines)

    def ask(self, session_id: str, question: str) -> dict:
        """
        Session-based chat:
        - keeps chat history temporarily
        - adds history into prompt
        """

        if not self.session_manager.session_exists(session_id):
            raise ValueError("Invalid session_id. Start a session first.")

        # Store user question in memory
        self.session_manager.add_message(session_id, "user", question)

        query_embedding = self.embedder.embed_query(question)
        retrieved_chunks = self.retriever.retrieve(query_embedding, top_k=self.top_k)

        if not retrieved_chunks:
            answer = "Not found in the knowledge base."

            # store assistant response
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
                "chunk_id": chunk["chunk_id"],
                "cosine_similarity": chunk["cosine_similarity"]
            })

        context = "\n\n".join(context_parts)

        # Get last few chat messages
        history = self.session_manager.get_history(session_id, last_n=6)
        chat_history_text = self._format_chat_history(history)

        prompt = build_rag_prompt(
            context=context,
            question=question,
            chat_history=chat_history_text
        )

        answer = self.llm.generate(prompt)

        # Store assistant response in memory
        self.session_manager.add_message(session_id, "assistant", answer)

        return {
            "answer": answer,
            "sources": sources
        }