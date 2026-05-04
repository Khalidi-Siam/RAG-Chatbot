import os
from pipeline.rag_pipeline import RAGPipeline


PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "uploads", "sample.pdf")


if __name__ == "__main__":
    rag = RAGPipeline(
        chunk_size=1000,
        chunk_overlap=150,
        similarity_threshold=0.6,
        top_k=5
    )

    print("=== Ingesting PDF ===")
    ingest_result = rag.ingest_pdf(PDF_PATH)
    print(ingest_result)

    print("\n=== Asking Question ===")
    question = "When first partition of Bengal happened?"
    response = rag.ask(question)

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources:")
    for src in response["sources"]:
        print(src)

    print("\n=== Asking Out-of-Scope Question ===")
    out_question = "Who won FIFA world cup 2022?"
    response2 = rag.ask(out_question)

    print("\nAnswer:")
    print(response2["answer"])

    print("\nSources:")
    print(response2["sources"])