from fastapi import FastAPI
from api.routes.session_routes import router as session_router
from api.routes.upload_routes import router as upload_router
from api.routes.chat_routes import router as chat_router

app = FastAPI(title="RAG Chatbot API")

app.include_router(session_router, prefix="/session", tags=["Session"])
app.include_router(upload_router, prefix="/session", tags=["Upload"])
app.include_router(chat_router, prefix="/session", tags=["Chat"])