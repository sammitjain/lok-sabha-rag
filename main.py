#!/usr/bin/env python3
"""Run the Lok Sabha RAG API server."""

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "lok_sabha_rag.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
