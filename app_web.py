# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:57:36 2026

@author: Samia
"""

import sys, os
import time
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, APIRouter, Query, Body, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# --- Imports from web_app_api/cairo_assistant ---
from cairo_assistant.chatbot_service import MAPS_DIR, process_chat_message, transcribe_audio_bytes
from live_location import (
    clear_tracked_live_location,
    get_live_location_payload,
    normalize_session_id,
    update_tracked_live_location,
)

# =========================
# Setup & Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
navitour_path = os.path.join(BASE_DIR.parent, "NaviTour")
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, navitour_path)

# =========================
# Database Logic
# =========================
def _candidate_database_urls():
    db_user = os.getenv("NAVITOUR_DB_USER", "postgres")
    db_password = os.getenv("NAVITOUR_DB_PASSWORD", "123456")
    db_host = os.getenv("NAVITOUR_DB_HOST", "localhost")
    db_port = os.getenv("NAVITOUR_DB_PORT", "5432")
    raw_names = os.getenv("NAVITOUR_DB_CANDIDATES", "egypt_transport,egypt_transport")
    db_names = [name.strip() for name in raw_names.split(",") if name.strip()]

    return [f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}" for db_name in db_names]

engine = None
DATABASE_URL = None

def get_engine():
    global engine, DATABASE_URL
    if engine is not None: return engine
    for url in _candidate_database_urls():
        candidate = create_engine(url, pool_pre_ping=True)
        try:
            with candidate.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine, DATABASE_URL = candidate, url
            return engine
        except: continue
    raise RuntimeError("Could not connect to database.")

# =========================
# Models
# =========================
class ChatPayload(BaseModel):
    message: str = ""

class RatingIn(BaseModel):
    user_id: int
    place_type: str        # "place" or "restaurant"
    item_id: int
    rating: int
    review: Optional[str] = ""

# =========================
# FastAPI Initialization
# =========================
app = FastAPI(title="NaviTour & Cairo Assistant Unified API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mounting Filesystems
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")
app.mount("/maps", StaticFiles(directory=str(MAPS_DIR)), name="maps")
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))

# =========================
# Combined Routes
# =========================

@app.get("/onboarding", response_class=HTMLResponse)
async def onboarding(request: Request):
    return templates.TemplateResponse(request=request, name="onboarding.html")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main Chat/Map Interface."""
    return templates.TemplateResponse(request=request, name="index.html")


# --- Chat & AI Endpoints (from web_app_api.py) ---

@app.post("/api/chat")
async def chat(payload: ChatPayload):
    response = process_chat_message(payload.message)
    status_code = 400 if response.get("error") else 200
    return JSONResponse(content=response, status_code=status_code)

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    extension = Path(audio.filename or "audio.webm").suffix or ".webm"
    transcript = transcribe_audio_bytes(audio_bytes, extension)
    if not transcript:
        return JSONResponse(content={"error": "Transcription failed."}, status_code=400)
    return {"text": transcript}

# --- Transport & Recommendations (from app.py) ---

base = APIRouter(prefix="/api", tags=["stations"])

@base.get("/stations")
def get_all_stations():
    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT station_id, name_ar AS name, line, ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon FROM metro_stations")).mappings().all()
    return [dict(r) for r in rows]

@base.get("/route")
def get_route(start: str, end: str, time: str = None):
    from raptor.services.raptor_service import run_raptor_from_assistant_json
    departure_time = time or datetime.now().strftime("%H:%M:%S")
    assistant_json = {"intent": "navigation", "start_point": {"official_name_ar": start}, "end_point": {"official_name_ar": end}}
    # Assuming 'network' is loaded globally as in your app.py
    result = run_raptor_from_assistant_json(network, assistant_json, departure_time)
    return {"route": result, "departure_time": departure_time}

app.include_router(base)

# --- User & Ratings Endpoints ---

@app.post("/api/users/register", tags=["users"])
def register_or_login(name: str = Body(...), password: str = Body(...)):
    # ... (Include the hashing and DB logic from your register_or_login function here)
    pass

@app.post("/api/ratings", tags=["ratings"])
def submit_rating(payload: RatingIn):
    # ... (Include the rating logic from your submit_rating function here)
    pass

# =========================
# Startup & Main
# =========================
@app.on_event("startup")
def startup():
    # Load RAPTOR network
    global network
    try:
        network_path = os.path.join(navitour_path, "data", "network.pkl")
        with open(network_path, "rb") as f:
            network = pickle.load(f)
        print("✅ RAPTOR network loaded")
    except Exception as e:
        print(f"❌ Network load failed: {e}")
        network = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)