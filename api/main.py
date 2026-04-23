"""
NaviTour — Unified Egypt Transport & Recommendation API
=======================================================
Single entry-point that replaces both app.py and app_web.py.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints summary:
  GET  /health
  GET  /                          → redirects to /onboarding (or serves navitour.html)

  # Chat & transcription
  POST /api/chat
  POST /api/transcribe
  POST /message                   (legacy dialogue endpoint — kept for chat widget)
  POST /reset
  POST /api/location
  POST /api/location/clear

  # Transport data
  GET  /api/stations
  GET  /api/stations/{name}/restaurants
  GET  /api/stations/{name}/places
  GET  /api/bus-stops
  GET  /api/metro-stations/geojson
  GET  /api/bus-stops/geojson
  GET  /api/bus-stops/{stop_id}/recommendations
  GET  /api/route

  # Users
  POST /api/users/register
  GET  /api/users/{user_id}
  POST /api/users/{user_id}/preferences

  # Ratings
  POST /api/ratings
  GET  /api/ratings/{user_id}
  GET  /api/item-ratings

  # Recommendations  (from recommender/router.py)
  GET  /api/recommend/{user_id}
  GET  /api/recommend/{user_id}/places
  GET  /api/recommend/{user_id}/restaurants
  POST /api/retrain
"""

# ─────────────────────────────────────────────────────────────
# stdlib / third-party
# ─────────────────────────────────────────────────────────────
import sys
import os
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI, APIRouter, Query, Body,
    HTTPException, Request, UploadFile, File,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────────────────────
# Path bootstrap
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
NAVITOUR_PATH = BASE_DIR.parent 

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(NAVITOUR_PATH))

# ─────────────────────────────────────────────────────────────
# Internal imports  (graceful fallback if modules are absent)
# ─────────────────────────────────────────────────────────────
try:
    from cairo_assistant.chatbot_service import (
        MAPS_DIR,
        build_map,
        process_chat_message,
        transcribe_audio_bytes,
    )
    _CHATBOT_AVAILABLE = True
except ImportError:
    MAPS_DIR = BASE_DIR / "maps"
    MAPS_DIR.mkdir(exist_ok=True)
    _CHATBOT_AVAILABLE = False
    def build_map(legs):
        return None
    def process_chat_message(msg):
        return {"assistant_message": "Chatbot service not available.", "error": False}
    def transcribe_audio_bytes(data, ext):
        return None

try:
    from live_location import (
        clear_tracked_live_location,
        get_live_location_payload,
        normalize_session_id,
        update_tracked_live_location,
    )
    _LIVE_LOC_AVAILABLE = True
except ImportError:
    _LIVE_LOC_AVAILABLE = False
    def normalize_session_id(sid): return sid or "default"
    def update_tracked_live_location(sid, lat, lon, acc=None): return normalize_session_id(sid)
    def clear_tracked_live_location(sid): pass
    def get_live_location_payload(sid, allow_fallback=False): return None


# ═══════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════

def _candidate_database_urls() -> list[str]:
    explicit = os.getenv("NAVITOUR_DATABASE_URL")
    if explicit:
        return [explicit]

    user     = os.getenv("NAVITOUR_DB_USER",     "postgres")
    password = os.getenv("NAVITOUR_DB_PASSWORD",  "123456")
    host     = os.getenv("NAVITOUR_DB_HOST",      "localhost")
    port     = os.getenv("NAVITOUR_DB_PORT",      "5432")
    raw      = os.getenv("NAVITOUR_DB_CANDIDATES","egypt_transport")
    names    = [n.strip() for n in raw.split(",") if n.strip()]
    return [f"postgresql://{user}:{password}@{host}:{port}/{name}" for name in names]


_engine       = None
_database_url = None


def get_engine():
    global _engine, _database_url
    if _engine is not None:
        return _engine
    last_err = None
    for url in _candidate_database_urls():
        candidate = create_engine(url, pool_pre_ping=True)
        try:
            with candidate.connect() as conn:
                conn.execute(text("SELECT 1"))
            _engine       = candidate
            _database_url = url
            print(f"✅  DB connected: {url.rsplit('/', 1)[-1]}")
            return _engine
        except Exception as exc:
            last_err = exc
    raise RuntimeError(
        "Could not connect to any NaviTour database. "
        "Set NAVITOUR_DATABASE_URL or NAVITOUR_DB_CANDIDATES."
    ) from last_err


def active_db_name() -> Optional[str]:
    return _database_url.rsplit("/", 1)[-1] if _database_url else None


def run_migrations():
    """Safe, idempotent schema updates applied at startup."""
    try:
        db = get_engine()
        with db.begin() as conn:
            conn.execute(text(
                "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS password TEXT"
            ))
            seq = conn.execute(text("SELECT to_regclass('users_user_id_seq')")).scalar()
            if seq:
                conn.execute(text("""
                    SELECT setval(
                        'users_user_id_seq',
                        GREATEST(COALESCE((SELECT MAX(user_id) FROM users), 1), 1),
                        true
                    )
                """))
        print("✅  Startup migrations applied")
    except Exception as exc:
        print(f"⚠️  Startup migrations skipped: {exc}")


def _table_exists(name: str) -> bool:
    with get_engine().connect() as conn:
        return bool(conn.execute(text("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = :n LIMIT 1
        """), {"n": name}).scalar())


def _metro_schema() -> bool:
    return _table_exists("metro_stations")


# ═══════════════════════════════════════════════════════════
# ROUTE HELPERS
# ═══════════════════════════════════════════════════════════

_route_stop_lookup    = None
_route_stop_name_func = None
network               = None   # populated at startup


def _get_stop_lookup():
    global _route_stop_lookup
    if _route_stop_lookup is None:
        lookup = {}
        if network is not None:
            for stop in network.stops.itertuples():
                lookup[str(stop.stop_id)] = {
                    "lat": float(stop.stop_lat),
                    "lon": float(stop.stop_lon),
                }
        _route_stop_lookup = lookup
    return _route_stop_lookup


def _get_stop_name_func():
    global _route_stop_name_func
    if _route_stop_name_func is None and network is not None:
        try:
            from raptor.output_translation import load_translations
            _route_stop_name_func = load_translations(
                str(NAVITOUR_PATH / "data" / "translations.txt"), network
            )
        except Exception:
            pass
    return _route_stop_name_func


def _serialize_leg(leg: dict) -> dict:
    lookup     = _get_stop_lookup()
    name_func  = _get_stop_name_func()

    raw_stops  = leg.get("stops") or [leg.get("from_stop"), leg.get("to_stop")]
    stop_names  = []
    stop_points = []
    for sid in raw_stops:
        if sid is None:
            continue
        sid   = str(sid)
        sname = name_func(sid) if name_func else sid
        stop_names.append(sname)
        coords = lookup.get(sid)
        if coords:
            stop_points.append({"stop_id": sid, "name": sname, **coords})

    from_id  = str(leg["from_stop"]) if leg.get("from_stop") is not None else None
    to_id    = str(leg["to_stop"])   if leg.get("to_stop")   is not None else None
    from_c   = lookup.get(from_id) if from_id else None
    to_c     = lookup.get(to_id)   if to_id   else None

    return {
        "mode":         leg.get("mode"),
        "agency":       leg.get("agency"),
        "route_short":  leg.get("route_short"),
        "route_long":   leg.get("route_long"),
        "trip_id":      leg.get("trip_id"),
        "shape_id":     leg.get("shape_id"),
        "from_stop_id": from_id,
        "to_stop_id":   to_id,
        "from_stop":    name_func(from_id) if name_func and from_id else from_id,
        "to_stop":      name_func(to_id)   if name_func and to_id   else to_id,
        "stops":        stop_names,
        "stop_points":  stop_points,
        "from_lat":     from_c["lat"] if from_c else None,
        "from_lon":     from_c["lon"] if from_c else None,
        "to_lat":       to_c["lat"]   if to_c   else None,
        "to_lon":       to_c["lon"]   if to_c   else None,
    }


def _serialize_route_option(option: dict, option_index: int = 0, recommended: bool = False) -> dict:
    raw_route = option.get("route")
    if raw_route is None:
        raw_route = option.get("legs") or []

    return {
        "option_index": option_index,
        "recommended": bool(recommended),
        "summary": option.get("summary", {}),
        "route": [_serialize_leg(leg) for leg in raw_route],
    }


def _build_serialized_route_payload(
    raw_legs,
    *,
    summary=None,
    start_name=None,
    destination_name=None,
    departure_time=None,
    route_options=None,
):
    serialized_options = []
    for index, option in enumerate(route_options or []):
        serialized_options.append(
            _serialize_route_option(
                option,
                option_index=index,
                recommended=(index == 0),
            )
        )

    return {
        "legs": [_serialize_leg(leg) for leg in (raw_legs or [])],
        "summary": summary or {},
        "start_name": start_name,
        "destination_name": destination_name,
        "departure_time": departure_time or (summary or {}).get("departure_time"),
        "route_options": serialized_options,
        "selected_option_index": 0 if serialized_options else None,
    }


# ═══════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════

class ChatPayload(BaseModel):
    message: str = ""


class MessagePayload(BaseModel):
    message:    str           = ""
    session_id: Optional[str] = None


class ResetPayload(BaseModel):
    session_id: Optional[str] = None


class LiveLocationIn(BaseModel):
    lat:        float
    lon:        float
    accuracy:   Optional[float] = None
    session_id: Optional[str]   = None


class LiveLocationClearIn(BaseModel):
    session_id: Optional[str] = None


class RatingIn(BaseModel):
    user_id:    int
    place_type: str           # "place" | "restaurant"
    item_id:    int
    rating:     int           # 1-5
    review:     Optional[str] = ""


# ═══════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title   = "NaviTour — Egypt Transport & Recommendation API",
    version = "3.0.0",
    description = "Unified backend for NaviTour (merged app.py + app_web.py).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Static files & templates
_STATIC_DIR    = PROJECT_ROOT / "web" / "static"
_TEMPLATE_DIR  = PROJECT_ROOT / "web" / "templates"

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
if MAPS_DIR.exists():
    app.mount("/maps", StaticFiles(directory=str(MAPS_DIR)), name="maps")

templates = Jinja2Templates(directory=str(_TEMPLATE_DIR)) if _TEMPLATE_DIR.exists() else None


# ─── Startup ────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    global network

    # Database migrations
    run_migrations()

    # Load RAPTOR transit network
    network_path = NAVITOUR_PATH / "data" / "network.pkl"
    try:
        with open(network_path, "rb") as fh:
            network = pickle.load(fh)
        print("✅  RAPTOR network loaded")
    except Exception as exc:
        print(f"⚠️  RAPTOR network not loaded: {exc}")
        network = None


# ─── Root ────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    # Serve navitour.html if it exists next to main.py, else redirect
    html_path = PROJECT_ROOT / "navitour.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return RedirectResponse(url="/onboarding", status_code=307)


@app.get("/onboarding", response_class=HTMLResponse, include_in_schema=False)
async def onboarding(request: Request):
    if templates:
        return templates.TemplateResponse(request=request, name="onboarding.html")
    return HTMLResponse("<h1>onboarding.html not found</h1>", status_code=404)


@app.get("/map.html", response_class=HTMLResponse, include_in_schema=False)
async def map_page(request: Request):
    if templates:
        return templates.TemplateResponse(request=request, name="map.html")
    return HTMLResponse("<h1>map.html not found</h1>", status_code=404)

@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def chat_page(request: Request):
    if templates:
        return templates.TemplateResponse(request=request, name="index.html")
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


# ═══════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════

@app.get("/health", tags=["system"])
def health():
    try:
        with get_engine().connect() as conn:
            if _metro_schema():
                counts = conn.execute(text("""
                    SELECT
                        (SELECT COUNT(*) FROM metro_stations) AS metro_stations,
                        (SELECT COUNT(*) FROM bus_stops)      AS bus_stops,
                        (SELECT COUNT(*) FROM places)         AS places,
                        (SELECT COUNT(*) FROM restaurants)    AS restaurants,
                        (SELECT COUNT(*) FROM users)          AS users,
                        (SELECT COUNT(*) FROM ratings)        AS ratings
                """)).mappings().one()
            else:
                counts = conn.execute(text("""
                    SELECT
                        (SELECT COUNT(*) FROM stations)    AS stations,
                        (SELECT COUNT(*) FROM places)      AS places,
                        (SELECT COUNT(*) FROM restaurants) AS restaurants,
                        (SELECT COUNT(*) FROM users)       AS users,
                        (SELECT COUNT(*) FROM ratings)     AS ratings
                """)).mappings().one()
        return {"status": "ok", "database": "connected", "db_name": active_db_name(), **dict(counts)}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


# ═══════════════════════════════════════════════════════════
# CHAT  (cairo_assistant)
# ═══════════════════════════════════════════════════════════

@app.post("/api/chat", tags=["chatbot"])
async def api_chat(payload: ChatPayload):
    result      = process_chat_message(payload.message)
    status_code = 400 if result.get("error") else 200
    return JSONResponse(content=result, status_code=status_code)


@app.post("/api/transcribe", tags=["chatbot"])
async def api_transcribe(audio: UploadFile = File(...)):
    data       = await audio.read()
    ext        = Path(audio.filename or "audio.webm").suffix or ".webm"
    transcript = transcribe_audio_bytes(data, ext)
    if not transcript:
        return JSONResponse({"error": "Transcription failed."}, status_code=400)
    return {"text": transcript}


# Legacy /message endpoint (used by chat widget)
_dialogue_sessions: dict = {}


@app.post("/message", tags=["chat"])
async def legacy_message(payload: MessagePayload):
    """
    Thin wrapper: tries the cairo_assistant chatbot first.
    Falls back to a simple echo if the service is unavailable.
    """
    sid   = normalize_session_id(payload.session_id)
    reply_data = process_chat_message(payload.message or "")

    reply      = (
        reply_data.get("assistant_message")
        or reply_data.get("reply")
        or "مرحبا! أنا مساعد NaviTour. كيف أقدر أساعدك؟"
    )
    route_payload = None
    if reply_data.get("route"):
        raw_route = reply_data["route"] or {}
        raw_legs = raw_route.get("legs") or raw_route.get("route") or []
        extractor_output = reply_data.get("extractor_output") or {}
        route_payload = _build_serialized_route_payload(
            raw_legs,
            summary={"legs_count": len(raw_legs)},
            start_name=extractor_output.get("start_point"),
            destination_name=extractor_output.get("end_point"),
        )
        route_payload["formatted_legs"] = raw_route.get("formatted_legs") or []
        route_payload["map_url"] = reply_data.get("map_url")

    live_location = get_live_location_payload(sid, allow_fallback=False)
    return {
        "reply":         reply,
        "session_id":    sid,
        "has_route":     route_payload is not None,
        "route":         route_payload,
        "live_location": live_location,
    }


@app.post("/reset", tags=["chat"])
async def legacy_reset(payload: Optional[ResetPayload] = Body(default=None)):
    sid = normalize_session_id(payload.session_id if payload else None)
    _dialogue_sessions.pop(sid, None)
    clear_tracked_live_location(sid)
    return {"status": "ok", "session_id": sid}


# ═══════════════════════════════════════════════════════════
# LIVE LOCATION
# ═══════════════════════════════════════════════════════════

@app.post("/api/location", tags=["location"])
def update_location(payload: LiveLocationIn):
    sid = update_tracked_live_location(
        payload.session_id, payload.lat, payload.lon, payload.accuracy
    )
    return {"status": "ok", "session_id": sid}


@app.post("/api/location/clear", tags=["location"])
def clear_location(payload: Optional[LiveLocationClearIn] = Body(default=None)):
    sid = normalize_session_id(payload.session_id if payload else None)
    clear_tracked_live_location(sid)
    return {"status": "ok", "session_id": sid}


# ═══════════════════════════════════════════════════════════
# TRANSPORT DATA  (router)
# ═══════════════════════════════════════════════════════════

transport = APIRouter(prefix="/api", tags=["transport"])


@transport.get("/stations")
def get_stations():
    with get_engine().connect() as conn:
        if _metro_schema():
            rows = conn.execute(text("""
                SELECT station_id,
                       name_ar AS name, name_en,
                       line, seq, is_interchange,
                       ST_Y(geom::geometry) AS lat,
                       ST_X(geom::geometry) AS lon
                FROM metro_stations
                ORDER BY line, seq
            """)).mappings().all()
        else:
            rows = conn.execute(text("""
                SELECT station_id, name, city, type,
                       ST_Y(geometry::geometry) AS lat,
                       ST_X(geometry::geometry) AS lon
                FROM stations ORDER BY name
            """)).mappings().all()
    return [dict(r) for r in rows]


@transport.get("/stations/{station_name}/restaurants")
def station_restaurants(
    station_name: str,
    distance_m: int = Query(800),
):
    with get_engine().connect() as conn:
        if _metro_schema():
            rows = conn.execute(text("""
                SELECT r.restaurant_id, r.name, r.cuisine,
                       ROUND(ST_Distance(s.geom::geography, r.geom::geography)) AS distance_m,
                       ST_Y(r.geom::geometry) AS lat, ST_X(r.geom::geometry) AS lon
                FROM metro_stations s
                JOIN restaurants r ON ST_DWithin(s.geom::geography, r.geom::geography, :d)
                WHERE (s.name_ar = :st OR s.name_en = :st) AND r.name IS NOT NULL
                ORDER BY distance_m LIMIT 10
            """), {"st": station_name, "d": distance_m}).mappings().all()
        else:
            rows = conn.execute(text("""
                SELECT r.restaurant_id, r.name, r.cuisine,
                       ROUND(ST_Distance(s.geometry::geography, r.geom::geography)) AS distance_m,
                       ST_Y(r.geom::geometry) AS lat, ST_X(r.geom::geometry) AS lon
                FROM stations s
                JOIN restaurants r ON ST_DWithin(s.geometry::geography, r.geom::geography, :d)
                WHERE s.name = :st AND r.name IS NOT NULL
                ORDER BY distance_m LIMIT 10
            """), {"st": station_name, "d": distance_m}).mappings().all()
    return {"station": station_name, "radius_m": distance_m, "count": len(rows),
            "restaurants": [dict(r) for r in rows]}


@transport.get("/stations/{station_name}/places")
def station_places(
    station_name: str,
    distance_m: int = Query(500),
):
    with get_engine().connect() as conn:
        if _metro_schema():
            rows = conn.execute(text("""
                SELECT p.place_id, p.name, p.category,
                       ROUND(ST_Distance(s.geom::geography, p.geom::geography)) AS distance_m,
                       ST_Y(p.geom::geometry) AS lat, ST_X(p.geom::geometry) AS lon
                FROM metro_stations s
                JOIN places p ON ST_DWithin(s.geom::geography, p.geom::geography, :d)
                WHERE (s.name_ar = :st OR s.name_en = :st) AND p.name IS NOT NULL
                ORDER BY distance_m LIMIT 10
            """), {"st": station_name, "d": distance_m}).mappings().all()
        else:
            rows = conn.execute(text("""
                SELECT p.place_id, p.name, p.category,
                       ROUND(ST_Distance(s.geometry::geography, p.geom::geography)) AS distance_m,
                       ST_Y(p.geom::geometry) AS lat, ST_X(p.geom::geometry) AS lon
                FROM stations s
                JOIN places p ON ST_DWithin(s.geometry::geography, p.geom::geography, :d)
                WHERE s.name = :st AND p.name IS NOT NULL
                ORDER BY distance_m LIMIT 10
            """), {"st": station_name, "d": distance_m}).mappings().all()
    return {"station": station_name, "radius_m": distance_m, "count": len(rows),
            "places": [dict(r) for r in rows]}


@transport.get("/bus-stops")
def get_bus_stops(
    lat:      Optional[float] = Query(None),
    lon:      Optional[float] = Query(None),
    radius_m: int             = Query(1000),
    agency_id:Optional[str]   = Query(None),
):
    if not _table_exists("bus_stops"):
        return []
    filters, params = [], {}
    if lat is not None and lon is not None:
        filters.append("ST_DWithin(b.geom::geography, ST_SetSRID(ST_MakePoint(:lon,:lat),4326)::geography, :radius)")
        params.update({"lat": lat, "lon": lon, "radius": radius_m})
    if agency_id:
        filters.append("b.agency_id = :agency")
        params["agency"] = agency_id
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    with get_engine().connect() as conn:
        rows = conn.execute(text(f"""
            SELECT b.stop_id, b.name_ar, b.name_en, b.agency_id,
                   b.near_metro, b.near_dist_m,
                   ST_Y(b.geom::geometry) AS lat, ST_X(b.geom::geometry) AS lon,
                   ms.name_ar AS metro_name_ar
            FROM bus_stops b
            LEFT JOIN metro_stations ms ON b.near_metro = ms.station_id
            {where} ORDER BY b.stop_id LIMIT 200
        """), params).mappings().all()
    return [dict(r) for r in rows]


@transport.get("/metro-stations/geojson")
def metro_geojson():
    if not _table_exists("metro_stations"):
        return {"type": "FeatureCollection", "features": []}
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT station_id, name_ar, name_en, line, seq,
                   ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon
            FROM metro_stations ORDER BY line, seq
        """)).mappings().all()
    features = []
    for r in rows:
        color = "#e74c3c" if r["line"] == "L1" else "#3498db"
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {**dict(r), "color": color, "type": "metro"},
        })
    return {"type": "FeatureCollection", "features": features}


@transport.get("/bus-stops/geojson")
def bus_stops_geojson():
    if not _table_exists("bus_stops"):
        return {"type": "FeatureCollection", "features": []}
    agency_colors = {
        "P_O_14": "#2ecc71", "CTA": "#ff6b35", "CTA_M": "#9b59b6",
        "P_B_8":  "#e67e22", "MM":  "#1abc9c", "COOP":  "#34495e",
        "GRN":    "#16a085", "BOX": "#8e44ad", "LTRA_M":"#c0392b",
    }
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT stop_id, name_ar, name_en, agency_id, near_metro, near_dist_m,
                   ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon
            FROM bus_stops ORDER BY stop_id
        """)).mappings().all()
    features = []
    for r in rows:
        color = agency_colors.get(r["agency_id"], "#95a5a6")
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {**dict(r), "color": color, "type": "bus"},
        })
    return {"type": "FeatureCollection", "features": features}


@transport.get("/bus-stops/{stop_id}/recommendations")
def bus_stop_recs(
    stop_id:  int,
    user_id:  str = Query("1"),
    radius_m: int = Query(1000),
):
    if not _table_exists("bus_stops"):
        raise HTTPException(404, "bus_stops table not found")
    with get_engine().connect() as conn:
        stop = conn.execute(text(
            "SELECT name_ar, name_en, geom FROM bus_stops WHERE stop_id = :id"
        ), {"id": stop_id}).mappings().first()
        if not stop:
            raise HTTPException(404, f"Bus stop {stop_id} not found")
        recs = conn.execute(text("""
            SELECT r.name, r.cuisine,
                   ROUND(ST_Distance(b.geom::geography, r.geom::geography)) AS distance_m
            FROM bus_stops b
            JOIN restaurants r ON ST_DWithin(b.geom::geography, r.geom::geography, :radius)
            WHERE b.stop_id = :stop_id
            ORDER BY distance_m LIMIT 15
        """), {"stop_id": stop_id, "radius": radius_m}).mappings().all()
    return {"stop_id": stop_id, "stop_name": stop["name_ar"],
            "radius_m": radius_m, "count": len(recs),
            "recommendations": [dict(r) for r in recs]}


@transport.get("/route")
def get_route(start: str, end: str, time: Optional[str] = None):
    if network is None:
        raise HTTPException(503, "Routing service not available")
    try:
        from raptor.services.raptor_service import run_raptor_from_assistant_json
        departure = time or datetime.now().strftime("%H:%M:%S")
        result = run_raptor_from_assistant_json(
            network,
            {"intent": "navigation",
             "start_point": {"official_name_ar": start},
             "end_point":   {"official_name_ar": end}},
            departure,
        )
        if isinstance(result, str):
            raise HTTPException(400, result)
        if isinstance(result, list):
            route_payload = _build_serialized_route_payload(
                result,
                summary={"legs_count": len(result)},
                start_name=start,
                destination_name=end,
                departure_time=departure,
            )
            return {
                "route": route_payload["legs"],
                "summary": route_payload["summary"],
                "departure_time": route_payload["departure_time"],
                "start_name": route_payload["start_name"],
                "destination_name": route_payload["destination_name"],
                "route_options": route_payload["route_options"],
                "selected_option_index": route_payload["selected_option_index"],
                "map_url": build_map(result),
            }
        if isinstance(result, dict):
            raw_legs = result.get("legs") or []
            route_payload = _build_serialized_route_payload(
                raw_legs,
                summary=result.get("summary", {}),
                start_name=start,
                destination_name=end,
                departure_time=departure,
                route_options=result.get("route_options") or [],
            )
            return {
                "route": route_payload["legs"],
                "summary": route_payload["summary"],
                "departure_time": route_payload["departure_time"],
                "start_name": route_payload["start_name"],
                "destination_name": route_payload["destination_name"],
                "route_options": route_payload["route_options"],
                "selected_option_index": route_payload["selected_option_index"],
                "map_url": build_map(raw_legs) if raw_legs else None,
            }
        raise HTTPException(500, f"Unexpected RAPTOR output: {type(result)}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Routing error: {exc}") from exc


app.include_router(transport)


# ═══════════════════════════════════════════════════════════
# USERS
# ═══════════════════════════════════════════════════════════

def _hash_pw(pw: str, salt: Optional[str] = None) -> str:
    if salt is None:
        salt = os.urandom(16).hex()
    return f"{salt}:{hashlib.sha256((salt + pw).encode()).hexdigest()}"


def _verify_pw(pw: str, stored: str) -> bool:
    try:
        salt = stored.split(":")[0]
        return _hash_pw(pw, salt) == stored
    except Exception:
        return False


@app.post("/api/users/register", tags=["users"])
def register_or_login(name: str = Body(...), password: str = Body(...)):
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name cannot be empty")
    if not password:
        raise HTTPException(400, "Password cannot be empty")
    try:
        with get_engine().connect() as conn:
            existing = conn.execute(
                text("SELECT user_id, name, password FROM users WHERE LOWER(name) = LOWER(:n)"),
                {"n": name},
            ).mappings().first()

        if existing:
            uid    = existing["user_id"]
            stored = existing["password"]
            if stored:
                if not _verify_pw(password, stored):
                    raise HTTPException(401, "Wrong password. Try again.")
            else:
                # First time setting password
                with get_engine().begin() as conn:
                    conn.execute(
                        text("UPDATE users SET password = :pw WHERE user_id = :uid"),
                        {"pw": _hash_pw(password), "uid": uid},
                    )
            return {"status": "ok", "user_id": existing["user_id"], "name": existing["name"],
                    "is_new": False, "message": f"Welcome back, {existing['name']}!"}

        with get_engine().begin() as conn:
            row = conn.execute(
                text("INSERT INTO users (name, password) VALUES (:n, :pw) RETURNING user_id"),
                {"n": name, "pw": _hash_pw(password)},
            ).fetchone()
        return {"status": "ok", "user_id": row[0], "name": name,
                "is_new": True, "message": f"Welcome, {name}!"}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Error: {exc}") from exc


@app.get("/api/users/{user_id}", tags=["users"])
def get_user(user_id: int):
    with get_engine().connect() as conn:
        user = conn.execute(
            text("SELECT user_id, name FROM users WHERE user_id = :uid"), {"uid": user_id}
        ).mappings().first()
    if not user:
        raise HTTPException(404, f"User {user_id} not found")
    return dict(user)


@app.post("/api/users/{user_id}/preferences", tags=["users"])
def save_preferences(
    user_id:   int,
    cuisines:  list = Body(default=[]),
    place_cats:list = Body(default=[]),
    radius:    int  = Body(default=1000),
):
    with get_engine().connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM ratings WHERE user_id = :uid"), {"uid": user_id}
        ).scalar()
    if count > 0:
        return {"status": "skipped", "message": "User already has ratings"}

    seeds = []
    with get_engine().connect() as conn:
        for c in cuisines:
            rows = conn.execute(
                text("SELECT restaurant_id FROM restaurants WHERE LOWER(cuisine) LIKE LOWER(:c) LIMIT 3"),
                {"c": f"%{c}%"},
            ).fetchall()
            seeds += [("restaurant", r[0]) for r in rows]
        for p in place_cats:
            rows = conn.execute(
                text("SELECT place_id FROM places WHERE LOWER(category) LIKE LOWER(:c) LIMIT 3"),
                {"c": f"%{p}%"},
            ).fetchall()
            seeds += [("place", r[0]) for r in rows]

    if not seeds:
        return {"status": "ok", "seeded": 0, "message": "No matching items found"}

    inserted = 0
    with get_engine().begin() as conn:
        for ptype, iid in seeds:
            exists = conn.execute(
                text("SELECT 1 FROM ratings WHERE user_id=:uid AND place_type=:pt AND place_id=:iid"),
                {"uid": user_id, "pt": ptype, "iid": iid},
            ).first()
            if not exists:
                conn.execute(
                    text("INSERT INTO ratings (user_id, place_type, place_id, rating, review) "
                         "VALUES (:uid, :pt, :iid, 5, 'Onboarding preference')"),
                    {"uid": user_id, "pt": ptype, "iid": iid},
                )
                inserted += 1

    return {"status": "ok", "seeded": inserted,
            "message": f"Saved {inserted} preference seeds — recommendations personalized!"}


# ═══════════════════════════════════════════════════════════
# RATINGS
# ═══════════════════════════════════════════════════════════

@app.post("/api/ratings", tags=["ratings"])
def submit_rating(payload: RatingIn):
    if not (1 <= payload.rating <= 5):
        raise HTTPException(400, "Rating must be 1–5")
    if payload.place_type not in ("place", "restaurant"):
        raise HTTPException(400, "place_type must be 'place' or 'restaurant'")
    try:
        with get_engine().begin() as conn:
            if not conn.execute(
                text("SELECT user_id FROM users WHERE user_id = :uid"), {"uid": payload.user_id}
            ).first():
                raise HTTPException(404, f"User {payload.user_id} not found")

            tbl = "places" if payload.place_type == "place" else "restaurants"
            col = "place_id" if payload.place_type == "place" else "restaurant_id"
            if not conn.execute(
                text(f"SELECT {col} FROM {tbl} WHERE {col} = :iid"), {"iid": payload.item_id}
            ).first():
                raise HTTPException(404, f"{payload.place_type.capitalize()} {payload.item_id} not found")

            existing = conn.execute(
                text("SELECT rating_id FROM ratings WHERE user_id=:uid AND place_type=:pt AND place_id=:iid"),
                {"uid": payload.user_id, "pt": payload.place_type, "iid": payload.item_id},
            ).first()

            if existing:
                conn.execute(
                    text("UPDATE ratings SET rating=:r, review=:rev, created_at=NOW() "
                         "WHERE user_id=:uid AND place_type=:pt AND place_id=:iid"),
                    {"uid": payload.user_id, "pt": payload.place_type,
                     "iid": payload.item_id, "r": payload.rating, "rev": payload.review or ""},
                )
                return {"status": "ok", "action": "updated", "message": "Rating updated!",
                        **payload.dict()}
            else:
                row = conn.execute(
                    text("INSERT INTO ratings (user_id, place_type, place_id, rating, review) "
                         "VALUES (:uid, :pt, :iid, :r, :rev) RETURNING rating_id"),
                    {"uid": payload.user_id, "pt": payload.place_type,
                     "iid": payload.item_id, "r": payload.rating, "rev": payload.review or ""},
                ).fetchone()
                return {"status": "ok", "action": "created", "rating_id": row[0],
                        "message": "Rating submitted!", **payload.dict()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Error: {exc}") from exc


@app.get("/api/ratings/{user_id}", tags=["ratings"])
def get_user_ratings(user_id: int):
    with get_engine().connect() as conn:
        if not conn.execute(
            text("SELECT user_id FROM users WHERE user_id = :uid"), {"uid": user_id}
        ).first():
            raise HTTPException(404, f"User {user_id} not found")
        rows = conn.execute(text("""
            SELECT rating_id, user_id, place_type,
                   place_id AS item_id, rating, review, created_at
            FROM ratings WHERE user_id = :uid ORDER BY created_at DESC
        """), {"uid": user_id}).mappings().all()
    return {"user_id": user_id, "total_ratings": len(rows), "ratings": [dict(r) for r in rows]}


@app.get("/api/item-ratings", tags=["ratings"])
def item_ratings(place_type: str, item_id: int):
    if place_type not in ("place", "restaurant"):
        raise HTTPException(400, "place_type must be 'place' or 'restaurant'")
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT rating_id, user_id, rating, review, created_at
            FROM ratings WHERE place_type = :pt AND place_id = :iid
            ORDER BY created_at DESC
        """), {"pt": place_type, "iid": item_id}).mappings().all()
    avg = round(sum(r["rating"] for r in rows) / len(rows), 2) if rows else 0
    return {"place_type": place_type, "item_id": item_id,
            "total_ratings": len(rows), "average_rating": avg,
            "ratings": [dict(r) for r in rows]}


# ═══════════════════════════════════════════════════════════
# RECOMMENDATION ROUTER  (from recommender/router.py)
# ═══════════════════════════════════════════════════════════
try:
    from recommender.router import router as rec_router, init_router
    init_router(get_engine)
    app.include_router(rec_router)
    print("✅  Recommender router loaded")
except Exception as _exc:
    print(f"⚠️  Recommender router not loaded: {_exc}")


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
