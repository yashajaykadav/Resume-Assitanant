import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import router, init_services

from app.database import Base,engine
from app import models

from app.middleware import RateLimitMiddleware , LoggingMiddleware


# ─── Logging setup ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ─── Startup / Shutdown ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        logger.warning("⚠️  GROQ_API_KEY is not set! Chat will fail. Add it to your .env file.")
    else:
        logger.info("✅  GROQ_API_KEY found.")
    logger.info("🚀  Starting Resume Chatbot API...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database table Created")
    init_services(settings)
    logger.info("✅  All services ready.")

    yield

    logger.info("👋  Shutting down.")


# ─── App creation ─────────────────────────────────────────
app = FastAPI(
    title="AI Resume Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

# ─── CORS ─────────────────────────────────────────────────
# Allows React web app and Flutter app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware,calls_per_minute=30)
app.add_middleware(LoggingMiddleware)

# ─── Routes ───────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


# ─── Root redirect ────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "AI Resume Chatbot API is running!",
        "docs": "/docs",
        "health": "/api/v1/health"
    }
