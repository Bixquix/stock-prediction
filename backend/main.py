from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes_llm import router as insight_router
from backend.api.routes_market import router as market_router
from backend.api.routes_predict import router as predict_router
from backend.config import get_settings

settings = get_settings()
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Predicts next-trading-day direction for any stock/index/crypto "
        "ticker using an 8-model ML ensemble trained on the fly, plus "
        "optional LLM-generated commentary from OpenAI or Gemini. "
        "Educational project - not financial advice."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

app.include_router(market_router)
app.include_router(predict_router)
app.include_router(insight_router)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api")
def api_root():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "endpoints": {
            "market_history": "/market/{ticker}/history?days=180",
            "predict": "/predict/{ticker}?force_retrain=false",
            "insight": "/insight/{ticker}?provider=openai|gemini",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/providers")
def providers():
    return {
        "default_provider": settings.DEFAULT_LLM_PROVIDER,
        "available": {
            "openai": bool(settings.OPENAI_API_KEY),
            "gemini": bool(settings.GOOGLE_API_KEY),
        },
        "models": {
            "openai": settings.OPENAI_MODEL,
            "gemini": settings.GEMINI_MODEL,
        },
    }
