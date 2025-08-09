# Image Analyze API (FastAPI)

Deploy-ready API for **EXIF extraction**, **palette/lighting/camera guess**, and optional **Google Vision** labels.
Use it as the backend for your GPT Action "Image Upload & Extract".

## Quickstart (Local)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # set ANALYZE_API_KEY in .env
uvicorn main:app --reload --port 8000
# Open http://localhost:8000/docs
```

## Docker

```bash
docker build -t image-analyze-api:latest .
docker run -p 8000:8000 -e ANALYZE_API_KEY=yourkey image-analyze-api:latest
```

## Cloud Deploy
Works great on Railway, Render, Fly.io, or any container platform.
- Set env var `ANALYZE_API_KEY`.
- (Optional) To enable Google Vision: set `USE_GOOGLE_VISION=1` and provide credentials per provider docs.

## GPT Actions
Use `openapi.yaml` with your deployed base URL.
Auth header: `x-api-key: <ANALYZE_API_KEY>`
