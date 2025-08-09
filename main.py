import io
import os
import logging
from typing import Optional, List, Dict, Any

from collections import Counter

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import exifread

# ---------------------------------------------------------
# App & Logging
# ---------------------------------------------------------
app = FastAPI(
    title="Image Analyze API",
    version="1.0.0",
    description="Receives an uploaded image (multipart/form-data) and returns technical + basic narrative analysis."
)

# Basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
app.logger = logger  # type: ignore[attr-defined]

# CORS (GPT Actions ve Postman testleri için açık)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # dilersen burada domain kısıtlayabilirsin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
ANALYZE_API_KEY = os.getenv("ANALYZE_API_KEY", "").strip()
USE_GOOGLE_VISION = os.getenv("USE_GOOGLE_VISION", "0").strip() in ("1", "true", "TRUE")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def open_image_from_upload(upload: UploadFile) -> Image.Image:
    """
    UploadFile içeriğini okur, PIL Image olarak döndürür.
    """
    content = upload.file.read()
    if not content:
        raise ValueError("Empty file content")

    # Reset pointer for other consumers if needed
    bio = io.BytesIO(content)
    img = Image.open(bio).convert("RGB")
    return img

def read_exif(content_bytes: bytes) -> Dict[str, Any]:
    """
    EXIF okumaya çalışır (PNG'lerde genelde yoktur). Hata olmazsa tag'leri döndürür.
    """
    tags: Dict[str, Any] = {}
    try:
        tags = exifread.process_file(io.BytesIO(content_bytes), details=False)
    except Exception:
        # PNG'lerde normal; sessizce geç
        pass
    return tags

def top3_palette(img: Image.Image) -> List[Dict[str, Any]]:
    """
    Hızlı ve bağımlılıksız baskın renkler: downscale + Counter (en çok görünen 3 renk).
    """
    arr = np.array(img)
    # downscale
    small = arr[::8, ::8, :].reshape(-1, 3)
    small_tuples = list(map(tuple, small.tolist()))
    common = Counter(small_tuples).most_common(3)
    return [{"rgb": list(rgb), "count": int(cnt)} for rgb, cnt in common]

def guess_format(filename: str) -> str:
    lower = (filename or "").lower()
    if lower.endswith(".png"):
        return "png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "jpg"
    if lower.endswith(".webp"):
        return "webp"
    if lower.endswith(".gif"):
        return "gif"
    return "unknown"

# (Opsiyonel) Google Vision
def run_google_vision_if_enabled(content_bytes: bytes) -> Dict[str, Any]:
    """
    USE_GOOGLE_VISION=1 ise Google Cloud Vision ile basit label/landmark/face özetleri alır.
    Hata olursa sessizce geçer.
    """
    result: Dict[str, Any] = {}
    if not USE_GOOGLE_VISION:
        return result
    try:
        from google.cloud import vision  # type: ignore
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content_bytes)

        # Label detection
        labels_resp = client.label_detection(image=image)
        labels = [f"{l.description} ({round(l.score,2)})" for l in labels_resp.label_annotations or []]

        # Landmark detection
        lm_resp = client.landmark_detection(image=image)
        landmarks = [f"{lm.description} ({round(lm.score,2)})" for lm in lm_resp.landmark_annotations or []]

        # Face detection (sadece count & likelihood)
        face_resp = client.face_detection(image=image)
        faces_info = []
        for f in face_resp.face_annotations or []:
            faces_info.append({
                "joy": str(f.joy_likelihood),
                "sorrow": str(f.sorrow_likelihood),
                "anger": str(f.anger_likelihood),
                "surprise": str(f.surprise_likelihood)
            })

        result = {
            "labels": labels[:10],
            "landmarks": landmarks[:5],
            "faces_likelihood": faces_info[:5]
        }
    except Exception as e:
        app.logger.info(f"Google Vision skipped or failed: {e}")
    return result

# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------
class HealthOut(BaseModel):
    status: str = "ok"

class AnalyzeOut(BaseModel):
    technical: Dict[str, Any]
    narrative: Dict[str, Any]
    warnings: List[str] = []
    vision: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/", response_model=HealthOut)
def root():
    return HealthOut(status="ok")

@app.get("/health", response_model=HealthOut)
def health():
    return HealthOut(status="ok")

@app.post("/analyze-image", response_model=AnalyzeOut)
async def analyze_image(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None)
):
    # API Key kontrolü
    if ANALYZE_API_KEY:
        if not x_api_key or x_api_key != ANALYZE_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

    # Hangi alan geldi?
    # (GPT bazen 'file', bazen 'image' alan adını kullanabiliyor.)
    upload = file or image
    # Ekstra: hangi alan adları gelmiş, loglayalım
    try:
        form = await request.form()
        app.logger.info(f"Form fields: {list(form.keys())}")
    except Exception:
        pass

    if upload is None or getattr(upload, "filename", None) is None:
        raise HTTPException(status_code=422, detail="No file found in form-data. Expected field 'file' (or 'image').")

    filename = upload.filename or ""
    fmt = guess_format(filename)

    # Dosyayı bytes olarak da alalım (EXIF & Vision için)
    try:
        content_bytes = await upload.read()
        if not content_bytes:
            raise ValueError("Empty file content")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    # PIL ile aç
    try:
        img = Image.open(io.BytesIO(content_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Unsupported image or read error: {e}")

    # EXIF
    warnings: List[str] = []
    tags = read_exif(content_bytes)
    if not tags:
        app.logger.info("PNG/JPG EXIF not found or minimal.")
        warnings.append("No EXIF or minimal EXIF (normal for many PNGs).")

    # Teknik bilgiler
    np_img = np.array(img)
    h, w = np_img.shape[:2]
    palette = top3_palette(img)

    technical = {
        "filename": filename,
        "format_guess": fmt,
        "resolution": {"width": int(w), "height": int(h)},
        "approx_palette_top3": palette,
        "exif_present": bool(tags),
    }

    # Basit anlatı (narrative) iskeleti — asıl derin analiz GPT tarafında yapılacak
    narrative = {
        "notes": "Basic technical cues extracted. Use these as grounding for your cinematic prompt builder.",
        "hints": [
            "If human subject: confirm age/gender/ethnicity & micro-imperfections.",
            "Confirm lighting type (natural/artificial), softness/hardness, direction.",
            "Confirm camera angle & focal length intent; add aperture for bokeh level.",
            "Describe environment style & texture (asphalt, neon, graffitis), imperfections.",
            "Add color palette as 60/30/10 using detected top colors + your taste."
        ]
    }

    # (Opsiyonel) Google Vision
    vision = run_google_vision_if_enabled(content_bytes)
    if vision:
        technical["vision_labels_count"] = len(vision.get("labels", []))

    return AnalyzeOut(
        technical=technical,
        narrative=narrative,
        warnings=warnings,
        vision=vision or None
    )
