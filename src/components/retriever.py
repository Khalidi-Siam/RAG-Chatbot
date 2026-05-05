from logger import logging

class Retriever:
    def __init__(self, vectorstore, similarity_threshold: float = 0.6):
        """
        similarity_threshold:
        - higher = stricter
        - cosine similarity range is typically 0.0 to 1.0
        """
        self.vectorstore = vectorstore
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Returns:
        [
            {
              "chunk_id": "...",
              "text": "...",
              "page": 2,
              "cosine_similarity": 0.73
            }
        ]
        """
        raw = self.vectorstore.query(query_embedding, top_k=top_k)

        if not raw:
            logging.warning("No results returned from vector store.")
            return []

        results = []

        for chunk_id, doc, meta, sim in zip(
            raw["ids"],
            raw["documents"],
            raw["metadatas"],
            raw["similarities"]
        ):
            if sim >= self.similarity_threshold:
                results.append({
                    "chunk_id": chunk_id,
                    "text": doc,
                    "page": meta.get("page"),
                    "source_file": meta.get("source_file", "unknown"),  # propagate source_file
                    "cosine_similarity": sim
                })

        results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        logging.info(f"Retrieved {len(results)} chunks above threshold {self.similarity_threshold}")
        return results