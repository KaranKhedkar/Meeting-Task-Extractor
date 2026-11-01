from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .core.database import init_db
from .api.routes import router
from .services.preprocessing import get_spacy_nlp
from .services.pipeline import get_action_classifier


# Ensure database tables exist
init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database schema
    init_db()
    # Pre-warm spaCy model and ActionClassifier
    get_spacy_nlp()
    get_action_classifier()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Meeting Task Extractor V1 — Classical NLP and Machine Learning for Action Item Extraction",
    lifespan=lifespan,
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow local frontend during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["system"])
def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "model": settings.SPACY_MODEL,
        "ml_model_loaded": get_action_classifier().ml_pipeline is not None,
    }
