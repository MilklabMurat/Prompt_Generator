# main.py
# Prompt-Generator Analyze API (full, from-scratch)
import io
import os
import logging
from typing import Optional, Dict, Any, Tuple, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageStat
import numpy as np
import exifread

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))  # 15MB
REQUIRE_API_KEY = bool(os.getenv("ANALYZE_API_KEY"))  # header kontrolü, varsa zorunlu
ALLOWED_MIME = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

USE_GOOGLE_VISION = os.getenv("USE_GOOGLE_VISION", "0") == "1"  # İstersek sonra açarız
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(
    title="Prompt-Generator Analyze API",
    version="1.0.0",
    description="Uploads bir görseli teknik + görsel analiz eder ve JSON döner.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # GPT Actions ve Swagger için açık
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Modeller
# -----------------------------------------------------------------------------
class AnalyzeResult(BaseModel):
    width: int
    height: int
    aspect_ratio: str
    mode: str
    approx_megapixels: float
    exif: Dict[str, Any]
    histogram_mean: float
    histogram_stddev: float
    brightness_estimate: float
    color_palette_hex: List[str]
    color_palette_ratio: List[float]


# -----------------------------------------------------------------------------
# Yardımcılar
# -----------------------------------------------------------------------------
def require_key_or_403(x_api_key: Optional[str]):
    if not REQUIRE_API_KEY:
        return
    must = os.getenv("ANALYZE_API_KEY", "").strip()
    if not must:
        # Ortam değişkeni yoksa gerektirme
        return
    if (x_api_key or "").strip() != must:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

def read_file_from_upload(upload: UploadFile) -> bytes:
    content = upload.file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Empty file.")
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_BYTES} bytes).")
    return content

def get_image_from_bytes(b: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(b))
        im.load()
        return im
    except Exception:
        raise HTTPException(status_code=415, detail="Unsupported image or corrupted bytes.")

def aspect_ratio_str(w: int, h: int) -> str:
    if h == 0:
        return "N/A"
    # sadeleştir
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    g = gcd(w, h)
    return f"{w//g}:{h//g}"

def try_read_exif(jpeg_like_bytes: bytes) -> Dict[str, Any]:
    # PNG'de EXIF yok; JPEG/JPG/WebP bazı durumlarda olur
    tags = {}
    try:
        stream = io.BytesIO(jpeg_like_bytes)
        stream.seek(0)
        tags_raw = exifread.process_file(stream, details=False)
        for k, v in tags_raw.items():
            tags[k] = str(v)
        if not tags:
            logger.info("PNG/JPG EXIF not found or minimal.")
    except Exception as e:
        logger.warning(f"EXIF read failed: {e}")
    return tags

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def simple_kmeans(arr: np.ndarray, k=3, iters=12, seed=42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basit KMeans (sklearn'siz) - arr: (N,3) uint8 -> centers:(k,3), labels:(N,)
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=k, replace=False)
    centers = arr[idx].astype(np.float32)

    for _ in range(iters):
        # L2 uzaklık
        dists = np.sqrt(((arr[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))  # (N,k)
        labels = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        for ci in range(k):
            mask = labels == ci
            if np.any(mask):
                new_centers[ci] = arr[mask].mean(axis=0)
            else:
                # boş kümeye denk gelirse rastgele ata
                new_centers[ci] = arr[rng.integers(0, arr.shape[0])]
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers.astype(np.uint8), labels

def dominant_palette(im: Image.Image, k=3) -> Tuple[List[str], List[float]]:
    """
    Küçült, RGB'ye çevir, KMeans ile palet çıkar.
    """
    thumb = im.copy().convert("RGB")
    thumb.thumbnail((320, 320))
    arr = np.asarray(thumb, dtype=np.uint8).reshape(-1, 3)
    centers, labels = simple_kmeans(arr, k=k)
    # oranlar
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    ratios = (counts / counts.sum()).tolist()
    # merkezleri parlaklıkça sıralayalım (opsiyonel)
    brightness = centers.mean(axis=1)
    order = np.argsort(-brightness)  # en parlak önce
    hexes = [rgb_to_hex(tuple(centers[i])) for i in order]
    ratios_sorted = [ratios[i] for i in order]
    return hexes, ratios_sorted

def basic_stats(im: Image.Image) -> Tuple[float, float, float]:
    """
    Histogram tabanlı ortalama, stddev, kabaca parlaklık tahmini (0-255).
    """
    gray = im.convert("L")
    stat = ImageStat.Stat(gray)
    mean = float(stat.mean[0])
    # stddev elde et
    # Pillow Stat'ta stddev yoksa var_yı kullan
    if hasattr(stat, "stddev") and stat.stddev:
        std = float(stat.stddev[0])
    else:
        var = float(stat.var[0]) if hasattr(stat, "var") and stat.var else 0.0
        std = var ** 0.5
    brightness = mean  # 0-255 skalası
    return mean, std, brightness

def pull_first_file_from_any_key(form: Dict[str, Any]) -> Tuple[Optional[str], Optional[UploadFile]]:
    """
    Form içindeki ilk dosya alanını yakalar (anahtar adı ne olursa olsun).
    """
    for k, v in form.items():
        # v: UploadFile tipinde mi?
        if hasattr(v, "filename") and v.filename:
            return k, v
    return None, None

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", tags=["root"])
def root():
    return {"ok": True, "message": "Prompt-Generator Analyze API"}

@app.get("/health", tags=["root"])
def health():
    return {"status": "healthy"}

@app.post("/echo-upload", tags=["debug"])
async def echo_upload(file: Optional[UploadFile] = File(default=None), x_api_key: Optional[str] = Header(default=None)):
    require_key_or_403(x_api_key)
    if file is None:
        # multipart geliyor mu teyit için Request.form() ile de bakabiliriz
        raise HTTPException(status_code=422, detail="No file uploaded (use 'file' field).")
    content = read_file_from_upload(file)
    return {
        "used_field": "file",
        "filename": file.filename,
        "content_type": file.content_type,
        "bytes": len(content),
    }

@app.post("/echo-anyfile", tags=["debug"])
async def echo_anyfile(request: Request, x_api_key: Optional[str] = Header(default=None)):
    require_key_or_403(x_api_key)
    form = await request.form()
    key, uf = pull_first_file_from_any_key(form)
    if uf is None:
        raise HTTPException(status_code=422, detail="No file found in multipart form-data")
    b = await uf.read()
    if not b:
        raise HTTPException(status_code=422, detail="Empty file")
    if len(b) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_BYTES} bytes)")
    return {
        "used_field": key,
        "filename": uf.filename,
        "content_type": getattr(uf, "content_type", None),
        "bytes": len(b),
    }

@app.post("/analyze-image", response_model=AnalyzeResult, tags=["analyze"])
async def analyze_image(
    request: Request,
    file: Optional[UploadFile] = File(default=None),   # standart ad
    image: Optional[UploadFile] = File(default=None),  # bazı istemciler 'image' gönderir
    x_api_key: Optional[str] = Header(default=None),
):
    require_key_or_403(x_api_key)

    # 1) Önce parametrelerden yakalamaya çalış
    picked: Optional[UploadFile] = file or image

    # 2) Bulunamazsa form'u tarayıp ilk dosyayı çek
    if picked is None:
        form = await request.form()
        _, picked = pull_first_file_from_any_key(form)

    if picked is None:
        raise HTTPException(status_code=422, detail="No file found. Send multipart/form-data with an image.")

    # Boyut ve tip kontrolü
    ctype = (picked.content_type or "").lower()
    # Not: Bazı istemciler ctype boş geçebilir, o yüzden sadece uyarı logla
    if ctype and ctype not in ALLOWED_MIME:
        logger.warning(f"Unusual content-type: {ctype} (allowed: {ALLOWED_MIME})")

    # Bytes oku
    raw = await picked.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Empty file bytes.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_BYTES} bytes).")

    # PIL Image
    im = get_image_from_bytes(raw)
    w, h = im.size

    # Basit istatistikler
    h_mean, h_std, brightness = basic_stats(im)

    # Renk paleti (3 renk)
    hexes, ratios = dominant_palette(im, k=3)

    # EXIF (varsa)
    exif = try_read_exif(raw)

    # (İleride) Google Vision kapalı
    if USE_GOOGLE_VISION:
        # Buraya Vision ile obje/detay analizi eklenebilir
        pass

    return AnalyzeResult(
        width=w,
        height=h,
        aspect_ratio=aspect_ratio_str(w, h),
        mode=str(im.mode),
        approx_megapixels=round((w * h) / 1_000_000.0, 3),
        exif=exif,
        histogram_mean=round(h_mean, 3),
        histogram_stddev=round(h_std, 3),
        brightness_estimate=round(brightness, 3),
        color_palette_hex=hexes,
        color_palette_ratio=[round(x, 4) for x in ratios],
    )
