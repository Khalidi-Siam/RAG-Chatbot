import os
import json
import numpy as np
import faiss


class FAISSVectorStore:
    def __init__(
        self,
        persist_path: str = "faiss_db",
        collection_name: str = "pdf_knowledge"
    ):
        self.persist_path = persist_path
        self.collection_name = collection_name

        self.index_path = os.path.join(persist_path, f"{collection_name}.index")
        self.meta_path = os.path.join(persist_path, f"{collection_name}_meta.json")

        os.makedirs(self.persist_path, exist_ok=True)

        self.index = None
        self.metadata = []  # aligned with FAISS internal index positions

        self._load()

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add_documents(self, chunks: list[dict], embeddings: list[list[float]]):
        """
        Store embeddings in FAISS index + metadata in JSON.
        Uses cosine similarity (via inner product on normalized vectors).
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks count and embeddings count mismatch!")

        embeddings_np = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings_np)

        dim = embeddings_np.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)  # cosine similarity (inner product)

        self.index.add(embeddings_np)

        for chunk in chunks:
            self.metadata.append({
                "id": chunk["chunk_id"],
                "document": chunk["text"],
                "metadata": {"page": chunk.get("page")}
            })

        self._save()

    def query(self, query_embedding: list[float], top_k: int = 5) -> dict:
        """
        Query FAISS and return cosine similarity scores.

        Returns format:
        {
          "ids": [...],
          "documents": [...],
          "metadatas": [...],
          "similarities": [...]
        }
        """
        if not query_embedding or self.index is None or self.index.ntotal == 0:
            return {}

        query_np = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_np)

        # D = similarity scores (cosine similarity)
        # I = indexes into metadata list
        D, I = self.index.search(query_np, top_k)

        ids_res = []
        docs_res = []
        metas_res = []
        sims_res = []

        for sim, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta_item = self.metadata[idx]

            ids_res.append(meta_item["id"])
            docs_res.append(meta_item["document"])
            metas_res.append(meta_item["metadata"])
            sims_res.append(float(sim))

        return {
            "ids": ids_res,
            "documents": docs_res,
            "metadatas": metas_res,
            "similarities": sims_res
        }

    def count(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def reset_collection(self):
        """
        Reset index and metadata.
        """
        self.index = None
        self.metadata = []

        if os.path.exists(self.index_path):
            os.remove(self.index_path)

        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)