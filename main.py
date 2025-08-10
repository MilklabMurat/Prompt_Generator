import io, os, logging
from typing import Optional, Dict, Any, Tuple, List

from fastapi import FastAPI, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
import numpy as np
import exifread

app = FastAPI(title="Image Analyze API", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

MAX_FILE_MB = 10
ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}
ENV_API_KEY = os.getenv("ANALYZE_API_KEY", "").strip()

def _allowed(filename: Optional[str]) -> bool:
    if not filename:
        return True  # bytes upload’ta ad gelmeyebilir; içerikten anlayacağız
    f = filename.lower()
    return any(f.endswith(x) for x in ALLOWED_EXTS)

def _load_pil(img_bytes: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(img_bytes))
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    return im

def _simple_kmeans(arr: np.ndarray, k: int = 3, iters: int = 12) -> np.ndarray:
    rng = np.random.default_rng(42)
    idx = rng.choice(arr.shape[0], size=min(k, arr.shape[0]), replace=False)
    centers = arr[idx].astype(np.float32)
    for _ in range(iters):
        d = ((arr[:, None, :].astype(np.float32) - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d, axis=1)
        new_centers = np.zeros_like(centers)
        for i in range(centers.shape[0]):
            m = labels == i
            new_centers[i] = arr[m].mean(axis=0) if np.any(m) else arr[rng.integers(0, arr.shape[0])]
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers.clip(0, 255).astype(np.uint8)

def _palette_hex(im: Image.Image, k: int = 3) -> List[str]:
    small = im.resize((128, 128))
    arr = np.array(small)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = arr.reshape(-1, 3)
    centers = _simple_kmeans(arr, k=k)
    return [f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}" for c in centers]

def _aspect_ratio(w: int, h: int) -> Tuple[str, float]:
    if h == 0: return "?", 0.0
    r = w / h
    target = [(1,1),(3,2),(4,3),(16,9),(2,1),(21,9)]
    best = min(target, key=lambda t: abs(r - (t[0]/t[1])))
    return f"{best[0]}:{best[1]}", r

def _brightness(im: Image.Image) -> float:
    arr = np.array(im.resize((128, 128)).convert("L"))
    return float(arr.mean()) / 255.0

@app.get("/", response_class=PlainTextResponse)
def root():
    return "OK"

@app.get("/health", response_class=JSONResponse)
def health():
    return {"status": "ok", "version": app.version}

@app.post("/analyze-image")
async def analyze_image(
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    file: bytes = File(..., description="PNG/JPG/JPEG dosyası (10MB)"),
):
    expected = ENV_API_KEY or "Milklab-AiStudio-Key-6455"
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")

    if not file or len(file) == 0:
        raise HTTPException(status_code=422, detail="Empty file body")

    if len(file) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_FILE_MB} MB)")

    # İçerik doğrulama (MIME sniffing)
    try:
        im = _load_pil(file)
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Unsupported image: {e}")

    w, h = im.size
    ar_label, ar_float = _aspect_ratio(w, h)
    palette = _palette_hex(im, k=3)
    bright = _brightness(im)

    # EXIF (JPEG varsa)
    try:
        tags = exifread.process_file(io.BytesIO(file), details=False)
        exif_data = {}
        for k in ("EXIF LensModel","EXIF FNumber","EXIF ExposureTime","EXIF ISOSpeedRatings",
                  "EXIF FocalLength","Image Model","EXIF DateTimeOriginal"):
            if k in tags: exif_data[k] = str(tags[k])
    except Exception:
        exif_data = {}

    return {
        "technical": {
            "width": w, "height": h, "mode": im.mode,
            "aspect_ratio": ar_label, "aspect_float": round(ar_float,4),
            "dominant_colors": palette,
            "avg_brightness_0_1": round(bright,4),
            "exif": exif_data
        },
        "narrative": {
            "subject_guess": "portrait" if h>w and bright<0.6 else "scene",
            "notes": ["Binary upload path used; Swagger shows real file picker."]
        },
        "warnings": [],
        "vision": {}
    }
