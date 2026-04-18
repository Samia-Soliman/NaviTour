# NaviTour Web Deployment

## Run locally

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Start the web app:

```powershell
python web_app.py
```

3. Open:

```text
http://127.0.0.1:5000
```

## What the web app does

- Sends the user message to `/api/chat`
- Predicts the intent using the local intent model
- If the intent is `navigation`, extracts `start_point` and `end_point` as JSON
- Runs the RAPTOR route engine
- Returns:
  - assistant text
  - extracted JSON
  - formatted route steps
  - generated route map link

## Notes before deployment

- `data/network.pkl` must exist on the server.
- The base Nile model must already be available in the Hugging Face cache because the loader uses `local_files_only=True`.
- The adapter is loaded from `cairo_assistant/nilechat_cairo_final_v1`.
- The intent model is loaded from `cairo_assistant/intent_model`.

## Production idea

For deployment on a VPS or Windows server, run the app behind a production WSGI server such as `waitress`:

```powershell
pip install waitress
waitress-serve --host 0.0.0.0 --port 5000 web_app:app
```
