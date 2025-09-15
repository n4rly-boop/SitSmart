from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.routes import router as api_router

app = FastAPI(title="SitSmart API", version="0.2.0")

# Basic CORS (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes under /api
app.include_router(api_router, prefix="/api")

# Serve static demo
static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", tags=["system"])  # Simple root for convenience
async def root():
    return {"name": "SitSmart API", "status": "ok"}


# Uvicorn entrypoint: `uvicorn app.main:app --reload`
