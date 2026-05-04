import os
from components.pdf_ingestor import PDFIngestor

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "uploads", "sample.pdf")

if __name__ == "__main__":
    ingestor = PDFIngestor(chunk_size=1000, chunk_overlap=150)

    result = ingestor.ingest(PDF_PATH)

    pages = result["pages"]
    chunks = result["chunks"]

    print(f"Total pages extracted: {len(pages)}")
    if pages:
        print("\n----- Page 1 Sample -----")
        print(pages[0]["text"][:500])

    print("\n=======================")
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print("\n----- Chunk Sample -----")
        print("Chunk ID:", chunks[0]["chunk_id"])
        print("Page:", chunks[0]["page"])
        print(chunks[0]["text"][:500])