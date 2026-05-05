import os
from pipeline.rag_pipeline import RAGPipeline


PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "uploads", "sample.pdf")


if __name__ == "__main__":
    rag = RAGPipeline()

    print("=== Ingesting PDF ===")
    ingest_result = rag.ingest_pdf(PDF_PATH)
    print(ingest_result)

    print("\n=== Starting Session ===")
    session_id = rag.start_session()
    print("Session ID:", session_id)

    print("\n=== Question 1 ===")
    q1 = "When Permanent Settlement Act was introduced?"
    res1 = rag.ask(session_id, q1)
    print("Answer:", res1["answer"])

    print("\n=== Follow-up Question (context dependent) ===")
    q2 = "What was the social impact of that act?"
    res2 = rag.ask(session_id, q2)
    print("Answer:", res2["answer"])

    print("\n=== Ending Session ===")
    rag.end_session(session_id)
    print("Session ended successfully.")