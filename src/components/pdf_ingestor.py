import os
import re
import uuid
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logging


class PDFIngestor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text.
        """
        if not text:
            return ""

        text = text.replace("\t", " ")
        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_pages(self, pdf_path: str) -> list[dict]:
        """
        Extract text page by page.

        Returns:
        [
            {"page": 1, "text": "..."},
            {"page": 2, "text": "..."}
        ]
        """
        logging.info(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        pages = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text")

            cleaned_text = self._clean_text(text)

            if cleaned_text:
                pages.append({
                    "page": page_index + 1,
                    "text": cleaned_text
                })
            else:
                logging.warning(f"Page {page_index + 1} is empty or unreadable - skipped.")

        doc.close()
        logging.info(f"Extracted {len(pages)} non-empty pages from '{os.path.basename(pdf_path)}'")
        return pages

    def chunk_pages(self, pages: list[dict], source_file: str) -> list[dict]:
        """
        Convert extracted pages into chunks.

        Returns:
        [
            {"chunk_id": "...", "page": 1, "text": "...", "source_file": "..."},
            {"chunk_id": "...", "page": 1, "text": "...", "source_file": "..."},
            {"chunk_id": "...", "page": 2, "text": "...", "source_file": "..."},
        ]
        """
        chunks = []

        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"]

            split_chunks = self.splitter.split_text(text)

            for chunk in split_chunks:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "page": page_num,
                    "text": chunk.strip(),
                    "source_file": source_file
                })

        return chunks

    def ingest(self, pdf_path: str) -> dict:
        """
        Full ingestion pipeline.

        Returns:
        {
            "pages": [...],
            "chunks": [...]
        }
        """
        source_file = os.path.basename(pdf_path)
        pages = self.extract_pages(pdf_path)
        chunks = self.chunk_pages(pages, source_file)

        logging.info(f"Ingestion complete: {len(pages)} pages -> {len(chunks)} chunks from '{source_file}'")
        return {
            "pages": pages,
            "chunks": chunks
        }