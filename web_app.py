from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from cairo_assistant.chatbot_service import MAPS_DIR, process_chat_message, transcribe_audio_bytes


BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "web" / "templates"),
    static_folder=str(BASE_DIR / "web" / "static"),
)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "")
    response = process_chat_message(message)
    status_code = 400 if response.get("error") else 200
    return jsonify(response), status_code


@app.post("/api/transcribe")
def transcribe():
    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({"error": "No audio file provided."}), 400

    audio_bytes = audio_file.read()
    transcript = transcribe_audio_bytes(audio_bytes, Path(audio_file.filename or "audio.webm").suffix or ".webm")
    if not transcript:
        return jsonify({"error": "Transcription failed."}), 400

    return jsonify({"text": transcript}), 200


@app.get("/maps/<path:filename>")
def serve_map(filename: str):
    return send_from_directory(MAPS_DIR, filename)


if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
