import io, os, logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import exifread

# ---------- Ayarlar ----------
API_KEY = os.getenv("ANALYZE_API_KEY", "").strip()
USE_GOOGLE = os.getenv("USE_GOOGLE_VISION", "0").strip() == "1"
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "10"))
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}  # HEIC yok
# -----------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

app = FastAPI(title="Image Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # GPT Actions sunucudan gelir, CORS engel olmasın
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Basit sağlık kontrolü ----
@app.get("/")
def root():
    return {"ok": True, "msg": "Image Analysis API up"}

# ---- Yardımcılar ----
def _get_file_from_form(possible_keys, request: Request, file_param: Optional[UploadFile]) -> UploadFile:
    """
    GPT Actions bazen form-data field adlarını farklı gönderebilir.
    Önce normal "file" paramı bakar; yoksa form üzerinden tüm olasıları dener.
    """
    if file_param is not None:
        return file_param

    # Starlette ile form'u kendimiz parse edelim:
    # NOT: Bu, requestBody multipart ise çalışır. Aksi 415 döneriz.
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise HTTPException(status_code=415, detail="Content-Type must be multipart/form-data")

    try:
        form = await_request_form(request)
        # await_request_form bir helper; aşağıda tanımlıyoruz.
        # Ama async fonksiyon; sync endpointte kullanamayız. Bu yüzden bu fonksiyonu sync'e sarmalıyız.
        # FastAPI sync endpointte form almak için low-level hack gerektirir.
        # Bunun yerine endpoint'i async yapacağız.
        raise RuntimeError("should not reach")  # bu satır sadece sync çağrıları yakalamak için
    except RuntimeError:
        # Bu fonksiyon sync endpointte çalışmayacak şekilde tasarlandı.
        # Endpoint'i async yapıp orada çağıracağız.
        pass

    # Buraya dönülmeyecek (endpoint async olacak).
    raise HTTPException(status_code=500, detail="Internal usage error")

async def await_request_form(request: Request):
    return await request.form()

def _ext_allowed(filename: str) -> bool:
    fname = filename or ""
    ext = os.path.splitext(fname)[1].lower()
    return ext in ALLOWED_EXT

def _enforce_size_limit(file_bytes: bytes):
    mb = len(file_bytes) / (1024 * 1024)
    if mb > MAX_FILE_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({mb:.2f} MB). Max {MAX_FILE_MB} MB")

def _read_exif_safely(file_bytes: bytes) -> Dict[str, Any]:
    # exifread sadece JPEG/TIFF’te anlamlı olur; PNG’de olmayabilir.
    try:
        tags = exifread.process_file(io.BytesIO(file_bytes), details=False)
        exif = {}
        for k, v in tags.items():
            exif[str(k)] = str(v)
        if not exif:
            log.info("PNG/JPG EXIF not found or minimal.")
        return exif
    except Exception:
        log.info("EXIF parse skipped/failed.")
        return {}

def _dominant_colors(image: Image.Image, k=3) -> Dict[str, Any]:
    """
    KMeans yerine hızlı ve deterministik bir yaklaşım: küçült & en sık renkler.
    (Render Free’de native kütüphanelerle uyumlu, hızlı, hatasız.)
    """
    im = image.convert("RGB").resize((64, 64))
    arr = np.array(im).reshape(-1, 3)
    # Basit histogram
    uniq, counts = np.unique(arr, axis=0, return_counts=True)
    order = np.argsort(-counts)
    top = uniq[order][:k]
    out = []
    for rgb in top:
        out.append({"rgb": rgb.tolist(), "hex": "#{:02x}{:02x}{:02x}".format(*rgb)})
    return {"palette_top": out}

def _narrative_stub(image: Image.Image) -> Dict[str, Any]:
    # Basit, güvenli, deterministik bir sahne/narratif tahmini (heuristic).
    w, h = image.size
    aspect = w / h if h else 1
    mood = "soft" if np.array(image).mean() > 128 else "moody"
    return {
        "scene_guess": "portrait/urban mix (heuristic)",
        "aspect_ratio": f"{w}x{h} ~ {aspect:.2f}:1",
        "mood_hint": mood,
        "notes": [
            "This is a lightweight heuristic summary. For deeper scene/subject analysis, enable Vision later."
        ],
    }

def _technical_from_pil(image: Image.Image, filename: str, exif: Dict[str, Any]) -> Dict[str, Any]:
    w, h = image.size
    mode = image.mode
    info = {
        "filename": filename,
        "width": w,
        "height": h,
        "channels": mode,
        "format": (os.path.splitext(filename)[1].lower() or "").lstrip("."),
        "exif": exif or None,
    }
    info.update(_dominant_colors(image))
    return info

# ---- Asıl endpoint (ASYNC ve sağlam) ----
@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
):
    # 1) Auth
    if API_KEY and (x_api_key or "").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")

    # 2) Multipart kontrolü
    ctype = (request.headers.get("content-type") or "").lower()
    if not ctype.startswith("multipart/form-data"):
        # GPT Action bazen “file yok ama çağırdım” diyebilir; net hata ver:
        raise HTTPException(status_code=415, detail="Content-Type must be multipart/form-data")

    # 3) Form’u oku, tüm olası anahtarları dene
    form = await request.form()
    uf: Optional[UploadFile] = None
    # Önce FastAPI'nin “file” paramını alalım (varsa)
    if file is not None:
        uf = file
    else:
        for key in ["file", "image", "image_file", "upload"]:
            if key in form:
                candidate = form[key]
                if isinstance(candidate, UploadFile):
                    uf = candidate
                    break

    if uf is None:
        # Form var ama dosya alanı yok → 422
        # GPT Action bu mesajı görünce UI’da dosya seçtirir, tekrar dener.
        raise HTTPException(status_code=422, detail="No file found in multipart form-data (expected fields: file / image / image_file / upload)")

    # 4) Dosya adı ve uzantı kontrol
    filename = uf.filename or "upload"
    if not _ext_allowed(filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file extension for {filename}. Allowed: {sorted(ALLOWED_EXT)}")

    # 5) Byte oku + boyut sınırı
    data = await uf.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    _enforce_size_limit(data)

    # 6) PIL ile oku
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # 7) EXIF (isteğe bağlı)
    exif = _read_exif_safely(data)

    # 8) Çıktılar
    technical = _technical_from_pil(im, filename, exif)
    narrative = _narrative_stub(im)

    # (İleride USE_GOOGLE true olursa vision alanını doldururuz)
    result = {
        "technical": technical,
        "narrative": narrative,
        "vision": None,
        "warnings": [],
    }

    return result
