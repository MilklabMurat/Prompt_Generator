import io
import os
import math
import logging
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
import numpy as np
import exifread

# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Image Analyze API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Actions + Swagger için açık
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MAX_FILE_MB = 10
ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}
ENV_API_KEY = os.getenv("ANALYZE_API_KEY", "").strip()  # Render'da set ettin

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _ext_ok(filename: str) -> bool:
    f = filename.lower()
    return any(f.endswith(x) for x in ALLOWED_EXTS)

def _read_image_bytes(upload: UploadFile) -> bytes:
    data = upload.file.read()
    if not data or len(data) == 0:
        raise HTTPException(status_code=422, detail="Empty file body")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_FILE_MB} MB)")
    return data

def _load_pil(img_bytes: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(img_bytes))
        # bazı PNG'ler 'P' modunda olabilir, dönüştürelim:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        return im
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Unsupported image: {e}")

def _safe_exif(img_bytes: bytes, filename: str) -> Dict[str, Any]:
    try:
        # EXIF sadece JPEG ağırlıklı; PNG'de genelde yok.
        if not filename.lower().endswith((".jpg", ".jpeg")):
            log.info("PNG/JPG EXIF not found or minimal.")
            return {}
        tags = exifread.process_file(io.BytesIO(img_bytes), details=False)
        wanted = ("EXIF LensModel", "EXIF FNumber", "EXIF ExposureTime",
                  "EXIF ISOSpeedRatings", "EXIF FocalLength", "Image Model",
                  "EXIF DateTimeOriginal")
        out = {}
        for k in wanted:
            if k in tags:
                out[k] = str(tags[k])
        return out
    except Exception:
        return {}

def _simple_kmeans(arr: np.ndarray, k: int = 3, iters: int = 12, seed: int = 42) -> np.ndarray:
    """
    Çok basit K-Means; scikit-learn bağımlılığı olmadan.
    arr: (N,3) uint8
    """
    if arr.shape[0] < k:
        return arr[:arr.shape[0]]
    rng = np.random.default_rng(seed)
    # Başlangıç merkezleri
    idx = rng.choice(arr.shape[0], size=k, replace=False)
    centers = arr[idx].astype(np.float32)

    for _ in range(iters):
        # atanım
        dists = ((arr[:, None, :].astype(np.float32) - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dists, axis=1)
        # yeni merkez
        new_centers = np.zeros_like(centers)
        for ci in range(k):
            mask = labels == ci
            if np.any(mask):
                new_centers[ci] = arr[mask].mean(axis=0)
            else:
                # boş küme: rastgele bir piksel ver
                new_centers[ci] = arr[rng.integers(0, arr.shape[0])]
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers.clip(0, 255).astype(np.uint8)

def _palette_hex(im: Image.Image, k: int = 3) -> List[str]:
    small = im.resize((128, 128))
    arr = np.array(small)
    if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA ise alpha'yı at
        arr = arr[:, :, :3]
    arr = arr.reshape(-1, 3)
    centers = _simple_kmeans(arr, k=k)
    hexes = ["#%02x%02x%02x" % (int(c[0]), int(c[1]), int(c[2])) for c in centers]
    return hexes

def _aspect_ratio(w: int, h: int) -> Tuple[str, float]:
    if h == 0:
        return "?", 0.0
    # Yakın oran adlandırması
    target = [(1,1),(3,2),(4,3),(16,9),(2,1),(21,9)]
    r = w / h
    best = min(target, key=lambda t: abs(r - (t[0]/t[1])))
    return f"{best[0]}:{best[1]}", r

def _brightness(im: Image.Image) -> float:
    arr = np.array(im.resize((128, 128)).convert("L"))
    return float(arr.mean()) / 255.0  # 0..1

# -----------------------------------------------------------------------------
# Health & Root
# -----------------------------------------------------------------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"

@app.get("/health", response_class=JSONResponse)
def health():
    return {"status": "ok", "version": app.version}

# -----------------------------------------------------------------------------
# Main endpoint
# -----------------------------------------------------------------------------
@app.post("/analyze-image")
async def analyze_image(
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    file: Optional[UploadFile] = File(None, description="Image file (PNG/JPG/JPEG)"),
    image: Optional[UploadFile] = File(None, description="Alternate field name"),
):
    # --- API KEY kontrolü
    expected = ENV_API_KEY or "Milklab-AiStudio-Key-6455"
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")

    # --- Dosyayı seç
    upload = file or image
    if upload is None:
        raise HTTPException(
            status_code=422,
            detail="Please send a file with key 'file' (or 'image') as multipart/form-data."
        )

    if not _ext_ok(upload.filename):
        raise HTTPException(status_code=415, detail=f"Only {sorted(ALLOWED_EXTS)} allowed")

    # --- Oku & PIL
    data = _read_image_bytes(upload)
    im = _load_pil(data)

    # --- Teknik özet
    w, h = im.size
    ar_label, ar_float = _aspect_ratio(w, h)
    palette = _palette_hex(im, k=3)
    bright = _brightness(im)
    exif = _safe_exif(data, upload.filename)

    technical = {
        "filename": upload.filename,
        "width": w,
        "height": h,
        "mode": im.mode,
        "aspect_ratio": ar_label,
        "aspect_float": round(ar_float, 4),
        "dominant_colors": palette,
        "avg_brightness_0_1": round(bright, 4),
        "exif": exif,
    }

    # --- Basit narrative (placeholder)
    subject = "portrait" if h > w and bright < 0.6 else "scene"
    narrative = {
        "subject_guess": subject,
        "notes": [
            "Colors computed with lightweight k-means (no sklearn).",
            "EXIF shown if present (mostly on JPEG)."
        ],
    }

    return {
        "technical": technical,
        "narrative": narrative,
        "warnings": [],
        "vision": {},
    }
