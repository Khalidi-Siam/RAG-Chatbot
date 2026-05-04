from components.pdf_ingestor import PDFIngestor
from components.embedder import GeminiEmbedder
from components.faiss_store import FAISSVectorStore
from components.retriever import Retriever
from components.gemini_llm import GeminiLLM
from components.prompt_template import build_rag_prompt


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

    def ingest_pdf(self, pdf_path: str):
        """
        PDF -> chunks -> embeddings -> FAISS store
        """
        ingestion_result = self.ingestor.ingest(pdf_path)
        chunks = ingestion_result["chunks"]

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_documents(texts)

        self.vectorstore.add_documents(chunks, embeddings)

        return {
            "total_pages": len(ingestion_result["pages"]),
            "total_chunks": len(chunks)
        }

    def ask(self, question: str) -> dict:
        """
        Query -> retrieve chunks -> build prompt -> LLM answer
        """
        query_embedding = self.embedder.embed_query(question)

        retrieved_chunks = self.retriever.retrieve(query_embedding, top_k=self.top_k)

        # fallback if nothing found
        if not retrieved_chunks:
            return {
                "answer": "Not found in the knowledge base.",
                "sources": []
            }

        # Build context from retrieved chunks
        context_parts = []
        sources = []

        for chunk in retrieved_chunks:
            context_parts.append(
                f"(Page {chunk['page']}) {chunk['text']}"
            )
            sources.append({
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "cosine_similarity": chunk["cosine_similarity"]
            })

        context = "\n\n".join(context_parts)

        prompt = build_rag_prompt(context=context, question=question)

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": sources
        }