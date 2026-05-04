import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    def __init__(
        self,
        persist_path: str = "data/chroma_db",
        collection_name: str = "pdf_knowledge"
    ):
        self.persist_path = persist_path
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=self.persist_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # IMPORTANT: Use cosine similarity metric
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # cosine distance
        )

    def add_documents(self, chunks: list[dict], embeddings: list[list[float]]):
        """
        Store documents into ChromaDB.

        chunks format:
        [
            {"chunk_id": "...", "page": 1, "text": "..."},
            ...
        ]
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks count and embeddings count mismatch!")

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]

        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "page": chunk.get("page"),
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> dict:
        """
        Query ChromaDB.

        Returns raw results including cosine distances.
        """
        if not query_embedding:
            return {}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "ids"]
        )

        return results

    def count(self) -> int:
        return self.collection.count()

    def reset_collection(self):
        """
        Delete and recreate collection.
        """
        self.client.delete_collection(self.collection_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )