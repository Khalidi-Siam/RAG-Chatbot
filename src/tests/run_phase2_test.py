import os
from components.pdf_ingestor import PDFIngestor
from components.embedder import GeminiEmbedder

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "uploads", "sample.pdf")


if __name__ == "__main__":
    # Step 1: Ingest PDF -> chunks
    ingestor = PDFIngestor(chunk_size=1000, chunk_overlap=150)
    result = ingestor.ingest(PDF_PATH)

    chunks = result["chunks"]
    chunk_texts = [c["text"] for c in chunks]

    print(f"Total chunks extracted: {len(chunk_texts)}")

    # Step 2: Initialize Gemini embedder
    embedder = GeminiEmbedder(model_name="models/gemini-embedding-001")

    # Step 3: Embed first few chunks (avoid large batch for testing)
    sample_texts = chunk_texts[:5]
    embeddings = embedder.embed_documents(sample_texts)

    print("\n=== Document Embedding Test ===")
    print(f"Total embeddings generated: {len(embeddings)}")

    if embeddings:
        print(f"Embedding dimension: {len(embeddings[0])}")
        print("First embedding preview (first 10 values):")
        print(embeddings[0][:10])

    # Step 4: Query embedding test
    query = "What is this document about?"
    query_embedding = embedder.embed_query(query)

    print("\n=== Query Embedding Test ===")
    print(f"Query embedding dimension: {len(query_embedding)}")
    print("Query embedding preview (first 10 values):")
    print(query_embedding[:10])

    # Step 5: Check dimension match
    if embeddings and query_embedding:
        assert len(embeddings[0]) == len(query_embedding), "Mismatch: document embedding dim != query embedding dim"

    print("\n✅ GeminiEmbedder test passed successfully.")