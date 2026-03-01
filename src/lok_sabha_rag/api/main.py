"""FastAPI application for Lok Sabha RAG."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from lok_sabha_rag.api.routes import search, synthesize, members, stats, question_text

app = FastAPI(
    title="Lok Sabha RAG",
    description="Search and synthesize answers from Lok Sabha Q&A documents",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(synthesize.router, prefix="/api", tags=["synthesize"])
app.include_router(members.router, prefix="/api", tags=["members"])
app.include_router(stats.router, prefix="/api", tags=["stats"])
app.include_router(question_text.router, prefix="/api", tags=["question-text"])

FRONTEND_DIR = Path(__file__).parent.parent.parent.parent / "frontend"


@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

