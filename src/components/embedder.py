import os
from google import genai
from google.genai import types

class GeminiEmbedder:
    def __init__(self, api_key: str = None, model_name: str = "gemini-embedding-001"):
        """
        model_name options:
        - gemini-embedding-001
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API Key not found.")
        
        # Initialize the new Client
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts using 'RETRIEVAL_DOCUMENT'.
        """
        if not texts:
            return []

        # The new SDK uses client.models.embed_content
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        
        # Return the list of embeddings
        return [item.values for item in result.embeddings]

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a single query using 'RETRIEVAL_QUERY'.
        """
        if not query or not query.strip():
            return []

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )
        
        # result.embeddings is a list even for single inputs
        return result.embeddings[0].values