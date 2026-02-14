"""
Faulter Core — Predictive Maintenance Backend
==============================================

FastAPI entry point.
Start with:  uvicorn main:app --reload
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Faulter Core",
    description=(
        "Predictive maintenance backend.  Translates mechanical stressors "
        "into electrical Current Signatures and detects faults by comparing "
        "live sensor data against a stored signature library."
    ),
    version="0.1.0",
)

# CORS — allow the React Flow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "Faulter Core",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }
