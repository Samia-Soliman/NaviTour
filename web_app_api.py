from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from cairo_assistant.chatbot_service import MAPS_DIR, process_chat_message, transcribe_audio_bytes

# Setup paths
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

# Mount Static Files (Equivalent to Flask's static_folder)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")

# Mount Maps Directory (Equivalent to Flask's send_from_directory route)
app.mount("/maps", StaticFiles(directory=str(MAPS_DIR)), name="maps")

# Setup Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))

# Pydantic model for Chat payload validation
class ChatPayload(BaseModel):
    message: str = ""

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Pass 'request' and 'name' explicitly as keywords
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.post("/api/chat")
async def chat(payload: ChatPayload):
    response = process_chat_message(payload.message)
    status_code = 400 if response.get("error") else 200
    return JSONResponse(content=response, status_code=status_code)

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # Read bytes from the uploaded file
    audio_bytes = await audio.read()
    
    # Get file extension or default to .webm
    extension = Path(audio.filename or "audio.webm").suffix or ".webm"
    
    transcript = transcribe_audio_bytes(audio_bytes, extension)
    
    if not transcript:
        return JSONResponse(content={"error": "Transcription failed."}, status_code=400)

    return {"text": transcript}

if __name__ == "__main__":
    import uvicorn
    # Use uvicorn to run the FastAPI app
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)