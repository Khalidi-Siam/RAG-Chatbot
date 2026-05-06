from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.session_routes import router as session_router
from api.routes.upload_routes import router as upload_router
from api.routes.chat_routes import router as chat_router
import os

app = FastAPI(title="RAG Chatbot API")

# Get CORS origins from environment variable
cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(session_router, prefix="/session", tags=["Session"])
app.include_router(upload_router, prefix="/session", tags=["Upload"])
app.include_router(chat_router, prefix="/session", tags=["Chat"])