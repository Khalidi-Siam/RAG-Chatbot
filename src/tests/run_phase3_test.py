import os
from components.pdf_ingestor import PDFIngestor
from components.embedder import GeminiEmbedder
from components.faiss_store import FAISSVectorStore
from components.retriever import Retriever


PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "uploads", "sample.pdf")


if __name__ == "__main__":
    ingestor = PDFIngestor(chunk_size=1000, chunk_overlap=150)
    ingestion_result = ingestor.ingest(PDF_PATH)

    chunks = ingestion_result["chunks"]
    print(f"Total chunks extracted: {len(chunks)}")

    embedder = GeminiEmbedder(model_name="gemini-embedding-001")

    chunks_to_store = chunks
    texts_to_store = [c["text"] for c in chunks_to_store]

    embeddings = embedder.embed_documents(texts_to_store)
    print(f"Generated embeddings: {len(embeddings)}")

    vectorstore = FAISSVectorStore(
        persist_path="faiss_db",
        collection_name="test_pdf_collection"
    )

    vectorstore.reset_collection()
    vectorstore.add_documents(chunks_to_store, embeddings)

    print(f"Stored chunks in FAISS: {vectorstore.count()}")

    query = "When first partition of Bengal happened?"
    query_embedding = embedder.embed_query(query)

    retriever = Retriever(vectorstore=vectorstore, similarity_threshold=0.5)

    retrieved_chunks = retriever.retrieve(query_embedding, top_k=5)

    print("\n=== Retrieved Chunks ===")
    print(f"Total retrieved: {len(retrieved_chunks)}")

    for i, item in enumerate(retrieved_chunks, start=1):
        print(f"\nResult {i}:")
        print("Chunk ID:", item["chunk_id"])
        print("Page:", item["page"])
        print("Cosine Similarity:", round(item["cosine_similarity"], 4))
        print("Text Preview:", item["text"][:300])