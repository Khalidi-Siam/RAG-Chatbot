from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    google_api_key: str
    database_url: str

    # Model names
    embedding_model: str = "gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    similarity_threshold: float = 0.6
    top_k: int = 5

    # Paths
    faiss_base_dir: str = "faiss_db"
    faiss_collection_name: str = "pdf_knowledge"
    upload_dir: str = "uploads"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # GOOGLE_API_KEY == google_api_key
        extra="ignore"          # ignore unknown env vars
    )


settings = Settings()  # module-level singleton
