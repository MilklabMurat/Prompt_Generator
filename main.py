# main.py
import io
import os
import sys
import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import exifread

# ------------------------------------------------------
# Logging
# ------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("app")

# ------------------------------------------------------
# FastAPI app & CORS
# ------------------------------------------------------
app = FastAPI(title="Prompt Generator Analyze API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # İstersen domain kısıtlayabilirsin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.responses import HTMLResponse

@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return """
    <html><head><meta charset="utf-8"><title>Privacy Policy</title></head>
    <body style="font-family:system-ui;max-width:800px;margin:40px auto;line-height:1.6">
      <h1>Privacy Policy</h1>
      <p>This service receives images you upload to analyze technical features (EXIF, colors, labels) and returns structured JSON.</p>
      <p>We do not sell your data. Uploaded files are processed in-memory and not retained longer than needed to fulfill the request.</p>
      <p>API access is protected via an API key. Do not share your key publicly.</p>
      <p>Contact: <a href="mailto:you@example.com">you@example.com</a></p>
      <p>Last updated: 2025-08-09</p>
    </body></html>
    """

@app.get("/terms", response_class=HTMLResponse)
def terms():
    return """
    <html><head><meta charset="utf-8"><title>Terms of Use</title></head>
    <body style="font-family:system-ui;max-width:800px;margin:40px auto;line-height:1.6">
      <h1>Terms of Use</h1>
      <p>Use this API at your own risk. You must own the rights to any content you upload. No unlawful content.</p>
      <p>We may rate-limit or revoke access if abuse is detected. No guarantees of availability or fitness for a particular purpose.</p>
      <p>Contact: <a href="mailto: info@milklab.com">info@milklab.com</a></p>
      <p>Last updated: 2025-08-09</p>
    </body></html>
    """
# ------------------------------------------------------
# MODELS
# ------------------------------------------------------
class EXIFModel(BaseModel):
    camera: Optional[str] = None
    lens: Optional[str] = None
    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    iso: Optional[int] = None
    shutter: Optional[str] = None
    timestamp: Optional[str] = None

class ColorItem(BaseModel):
    hex: str
    rgb: Dict[str, int]
    name: str

class TechModel(BaseModel):
    exif: Optional[EXIFModel] = None
    palette: Optional[List[ColorItem]] = None
    vision_labels: Optional[List[str]] = None

class NarrativeModel(BaseModel):
    setting: Optional[str] = None
    subjects: Optional[List[str]] = None
    mood: Optional[str] = None
    actions: Optional[List[str]] = None

class AnalyzeResponse(BaseModel):
    status: str
    tech: TechModel
    narrative: NarrativeModel
    errors: Dict[str, str] = {}

# ------------------------------------------------------
# Helpers: EXIF
# ------------------------------------------------------
def _to_float(val) -> Optional[float]:
    try:
        return float(str(val))
    except Exception:
        return None

def _to_int(val) -> Optional[int]:
    try:
        return int(str(val))
    except Exception:
        return None

def extract_exif_from_bytes(content: bytes) -> EXIFModel:
    """
    EXIF bilgilerini bytes üzerinden okur. PNG'lerde çoğunlukla EXIF yoktur.
    """
    try:
        stream = io.BytesIO(content)
        tags = exifread.process_file(stream, details=False)

        # Bazı tipik alanlar
        camera = str(tags.get("Image Model") or "") or None
        lens = str(tags.get("EXIF LensModel") or "") or None

        focal = tags.get("EXIF FocalLength")
        aperture = tags.get("EXIF FNumber")
        iso = tags.get("EXIF ISOSpeedRatings") or tags.get("EXIF PhotographicSensitivity")
        shutter = tags.get("EXIF ExposureTime")
        dt = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")

        focal_val = None
        if focal:
            # "85/1" gibi gelebilir
            try:
                s = str(focal)
                if "/" in s:
                    num, den = s.split("/")
                    focal_val = float(num) / float(den)
                else:
                    focal_val = float(s)
            except Exception:
                focal_val = None

        aperture_val = None
        if aperture:
            try:
                s = str(aperture)
                if "/" in s:
                    num, den = s.split("/")
                    aperture_val = float(num) / float(den)
                else:
                    aperture_val = float(s)
            except Exception:
                aperture_val = None

        iso_val = _to_int(iso) if iso else None
        shutter_val = str(shutter) if shutter else None
        ts_val = str(dt) if dt else None

        # Boş ise None dönelim
        if not any([camera, lens, focal_val, aperture_val, iso_val, shutter_val, ts_val]):
            logger.info("PNG/JPG EXIF not found or minimal.")
            return EXIFModel()

        return EXIFModel(
            camera=camera,
            lens=lens,
            focal_length=focal_val,
            aperture=aperture_val,
            iso=iso_val,
            shutter=shutter_val,
            timestamp=ts_val,
        )
    except Exception as e:
        logger.exception("EXIF parse failed")
        return EXIFModel()

# ------------------------------------------------------
# Helpers: Palette (no sklearn)
# ------------------------------------------------------
def _approx_color_name(rgb):
    r, g, b = rgb
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r < 35 and g < 35 and b < 35:
        return "black"
    if r > g and r > b:
        return "red-ish"
    if g > r and g > b:
        return "green-ish"
    if b > r and b > g:
        return "blue-ish"
    if r > 180 and g > 180 and b < 100:
        return "yellow-ish"
    if r > 180 and b > 180 and g < 100:
        return "magenta-ish"
    if g > 180 and b > 180 and r < 100:
        return "cyan-ish"
    return "neutral"

def cluster_palette(img: Image.Image, k: int = 3) -> List[Dict]:
    """
    Pillow ADAPTIVE palette ile hafif palet çıkarımı.
    """
    small = img.convert("RGB").resize((128, 128))
    colors_target = max(k * 3, k)
    pal_img = small.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors_target)

    palette = pal_img.getpalette()  # [r0,g0,b0, r1,g1,b1, ...]
    counts = pal_img.getcolors() or []  # [(count, index), ...]

    def idx_to_rgb(idx: int):
        base = idx * 3
        return tuple(palette[base:base + 3])

    ranked = sorted(counts, key=lambda x: x[0], reverse=True)

    top = []
    seen = set()
    for count, idx in ranked:
        rgb = idx_to_rgb(idx)
        if rgb in seen:
            continue
        seen.add(rgb)
        hexcol = "#{:02x}{:02x}{:02x}".format(*rgb)
        top.append({
            "hex": hexcol,
            "rgb": {"r": rgb[0], "g": rgb[1], "b": rgb[2]},
            "name": _approx_color_name(rgb)
        })
        if len(top) >= k:
            break

    return top

# ------------------------------------------------------
# Helpers: Google Vision (optional)
# ------------------------------------------------------
def google_vision_labels_bytes(content: bytes) -> List[str]:
    """
    USE_GOOGLE_VISION=1 ise Google Vision ile label listesi döndürür.
    """
    if os.environ.get("USE_GOOGLE_VISION", "0") != "1":
        return []

    try:
        from google.cloud import vision  # lazy import
    except Exception as e:
        logger.warning(f"google-cloud-vision import failed: {e}")
        return []

    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content)
        resp = client.label_detection(image=image)
        if resp.error and resp.error.message:
            logger.error(f"Vision API error: {resp.error.message}")
            return []
        labels = [l.description for l in (resp.label_annotations or [])]
        return labels
    except Exception as e:
        logger.exception("Vision request failed")
        return []

# ------------------------------------------------------
# Helpers: Narrative (basit otomatik özet)
# ------------------------------------------------------
def build_narrative(img: Image.Image, tech: Dict) -> NarrativeModel:
    # Çok basit bir çıkarım. İstersen burada Vision label’ları, paleti, EXIF’i harmanlayıp
    # daha akıllı bir özet üretecek şekilde genişletebiliriz.
    labels = tech.get("vision_labels") or []
    palette = tech.get("palette") or []

    mood = None
    if palette:
        # kaba ton tahmini
        names = [c["name"] for c in palette]
        if "blue-ish" in names:
            mood = "cool / melancholic"
        elif "yellow-ish" in names:
            mood = "warm / hopeful"
        else:
            mood = "neutral"
    else:
        mood = "neutral"

    setting = None
    if labels:
        # ör: "Street", "Sky", "Person" gibi
        if "Street" in labels:
            setting = "urban street"
        elif "Sky" in labels:
            setting = "outdoor with sky"
        else:
            setting = "unspecified"
    else:
        setting = "unspecified"

    subjects = []
    if labels:
        if "Person" in labels:
            subjects.append("person")

    return NarrativeModel(
        setting=setting,
        subjects=subjects,
        mood=mood,
        actions=[]
    )

# ------------------------------------------------------
# Routes
# ------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Prompt Generator Analyze API up"}

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/analyze-image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...), x_api_key: str = Header(default=None)):
    # API Key kontrolü
    expected_key = os.environ.get("ANALYZE_API_KEY")
    if not expected_key:
        logger.warning("ANALYZE_API_KEY not set on server")
        raise HTTPException(status_code=500, detail="Server missing ANALYZE_API_KEY")
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Dosyayı oku + Pillow ile aç
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        logger.exception("Image read/open failed")
        raise HTTPException(status_code=400, detail=f"Bad image file: {type(e).__name__}: {e}")

    out_errors: Dict[str, str] = {}
    tech: Dict = {}
    narrative: Dict = {}

    # EXIF
    try:
        exif = extract_exif_from_bytes(content)
        tech["exif"] = exif.model_dump()
    except Exception as e:
        logger.exception("EXIF failed")
        out_errors["exif"] = f"{type(e).__name__}: {e}"

    # Palette
    try:
        pal = cluster_palette(img, k=3)
        tech["palette"] = pal
    except Exception as e:
        logger.exception("Palette failed")
        out_errors["palette"] = f"{type(e).__name__}: {e}"

    # Vision (opsiyonel)
    try:
        labels = google_vision_labels_bytes(content)
        if labels:
            tech["vision_labels"] = labels
    except Exception as e:
        logger.exception("Vision failed")
        out_errors["vision"] = f"{type(e).__name__}: {e}"

    # Narrative
    try:
        nar = build_narrative(img, tech)
        narrative = nar.model_dump()
    except Exception as e:
        logger.exception("Narrative failed")
        out_errors["narrative"] = f"{type(e).__name__}: {e}"

    status = "ok" if not out_errors else "partial_success"

    return AnalyzeResponse(
        status=status,
        tech=TechModel(**tech),
        narrative=NarrativeModel(**narrative),
        errors=out_errors
    )

# ------------------------------------------------------
# Run locally (optional)
# ------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
