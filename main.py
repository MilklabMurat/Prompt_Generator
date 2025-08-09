import io
import os
from typing import Optional, List, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import exifread
import numpy as np
from PIL import Image

def simple_kmeans(arr: np.ndarray, k=3, iters=12, seed=42):
    """
    arr: (N,3) float32 RGB array
    k: clusters
    returns: (k,3) int centers
    """
    rng = np.random.default_rng(seed)
    # init: rastgele k piksel seç
    idx = rng.choice(arr.shape[0], size=k, replace=False)
    centers = arr[idx].copy()  # (k,3)
    for _ in range(iters):
        # uzaklık ve atama
        dists = ((arr[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        # yeni merkezler
        new_centers = []
        for ci in range(k):
            pts = arr[labels == ci]
            if pts.size == 0:
                # boş küme: rasgele yeniden başlat
                new_centers.append(arr[rng.integers(0, arr.shape[0])])
            else:
                new_centers.append(pts.mean(axis=0))
        new_centers = np.stack(new_centers, axis=0)
        if np.allclose(new_centers, centers, atol=1e-3):
            centers = new_centers
            break
        centers = new_centers
    return centers.astype(int)

def cluster_palette(img: Image.Image, k=3) -> list[str]:
    """
    Görselden k adet baskın rengi isimlendirerek döndürür.
    """
    small = img.resize((128,128)).convert("RGB")
    arr = np.array(small).reshape(-1,3).astype(np.float32)
    centers = simple_kmeans(arr, k=k, iters=12, seed=42).tolist()

    def rgb_to_name(r,g,b):
        if r>200 and g>200 and b>200: return "soft white"
        if r<50 and g<50 and b<50: return "charcoal"
        if b>r and b>g: return "deep blue" if b>150 else "steel blue"
        if g>r and g>b: return "leaf green" if g>150 else "olive"
        if r>g and r>b: return "warm red" if r>150 else "rust"
        return f"rgb({r},{g},{b})"

    return [rgb_to_name(*c) for c in centers]

USE_GOOGLE_VISION = os.getenv("USE_GOOGLE_VISION", "0") == "1"
VISION_CLIENT = None
if USE_GOOGLE_VISION:
    try:
        from google.cloud import vision
        VISION_CLIENT = vision.ImageAnnotatorClient()
    except Exception:
        VISION_CLIENT = None

API_KEY = os.getenv("ANALYZE_API_KEY")

app = FastAPI(title="Image Analyze API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Lighting(BaseModel):
    type: str | None = None
    quality: str | None = None
    direction: str | None = None
    ratio: str | None = None

class CameraGuess(BaseModel):
    angle: str | None = None
    lens_guess_mm: int | None = None
    aperture_guess: str | None = None

class Technical(BaseModel):
    lighting: Lighting
    camera: CameraGuess
    palette_60_30_10: List[str] = []
    atmosphere: List[str] = []

class ExifData(BaseModel):
    camera_model: str | None = None
    lens: str | None = None
    focal_length_mm: float | None = None
    iso: int | None = None
    shutter_s: float | None = None
    aperture_f: float | None = None
    aspect_ratio: str | None = None

class Narrative(BaseModel):
    subject: str | None = None
    emotion: str | None = None
    environment: str | None = None
    action_main: str | None = None
    secondary_characters: List[str] = []
    props: List[str] = []
    interaction: str | None = None

class Analysis(BaseModel):
    exif: ExifData
    technical: Technical
    narrative: Narrative
    confidence: Dict[str, float] = {}

def require_key(x_api_key: Optional[str]):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def parse_exif_bytes(img_bytes: bytes) -> ExifData:
    try:
        tags = exifread.process_file(io.BytesIO(img_bytes), details=False)
    except Exception:
        tags = {}
    model = str(tags.get("Image Model", "")) or None
    lens = str(tags.get("EXIF LensModel", "")) or None

    focal = None
    fr = str(tags.get("EXIF FocalLength", "")).strip()
    if fr:
        if "/" in fr:
            num, den = fr.split("/", 1)
            try: focal = float(num)/float(den)
            except: focal = None
        else:
            try: focal = float(fr)
            except: focal = None

    iso = None
    iso_tag = tags.get("EXIF ISOSpeedRatings") or tags.get("EXIF PhotographicSensitivity")
    if iso_tag:
        try: iso = int(str(iso_tag))
        except: iso = None

    shutter = None
    st = tags.get("EXIF ExposureTime")
    if st:
        s = str(st)
        if "/" in s:
            num, den = s.split("/", 1)
            try: shutter = float(num)/float(den)
            except: shutter = None
        else:
            try: shutter = float(s)
            except: shutter = None

    aperture = None
    ft = tags.get("EXIF FNumber")
    if ft:
        s = str(ft)
        if "/" in s:
            num, den = s.split("/", 1)
            try: aperture = float(num)/float(den)
            except: aperture = None
        else:
            try: aperture = float(s)
            except: aperture = None

    aspect = None
    try:
        im = Image.open(io.BytesIO(img_bytes))
        w, h = im.size
        def gcd(a,b):
            while b: a,b = b,a%b
            return a
        g = gcd(w,h)
        aspect = f"{w//g}:{h//g}"
    except Exception:
        pass

    return ExifData(
        camera_model=model, lens=lens, focal_length_mm=focal, iso=iso,
        shutter_s=shutter, aperture_f=aperture, aspect_ratio=aspect
    )

def cluster_palette(img: Image.Image, k=3) -> List[str]:
    small = img.resize((128,128)).convert("RGB")
    arr = np.array(small).reshape(-1,3).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(arr)
    centers = km.cluster_centers_.astype(int).tolist()
    def rgb_to_name(r,g,b):
        if r>200 and g>200 and b>200: return "soft white"
        if r<50 and g<50 and b<50: return "charcoal"
        if b>r and b>g: return "deep blue" if b>150 else "steel blue"
        if g>r and g>b: return "leaf green" if g>150 else "olive"
        if r>g and r>b: return "warm red" if r>150 else "rust"
        return f"rgb({r},{g},{b})"
    return [rgb_to_name(*c) for c in centers]

def guess_lighting_and_camera(im: Image.Image):
    w, h = im.size
    arr = np.array(im.resize((256,256)).convert("L")).astype(np.float32)
    contrast = float(arr.std())
    quality = "hard" if contrast > 60 else "soft"
    direction = "ambient"
    ratio = "2:1" if quality=="soft" else "4:1"
    angle = "eye"
    if h/w > 1.3: angle = "closeup"
    return Lighting(type="mixed", quality=quality, direction=direction, ratio=ratio), CameraGuess(angle=angle)

def google_vision_labels(content: bytes) -> List[str]:
    if not (USE_GOOGLE_VISION and VISION_CLIENT):
        return []
    try:
        from google.cloud import vision
        image = vision.Image(content=content)
        resp = VISION_CLIENT.label_detection(image=image)
        return [l.description for l in resp.label_annotations]
    except Exception:
        return []

@app.post("/analyze-image", response_model=Analysis)
async def analyze_image(file: UploadFile, x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)
    b = await file.read()
    if not b or len(b) > 10*1024*1024:
        raise HTTPException(413, "Image too large or empty")

    try:
        im = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Unsupported image")

    exif = parse_exif_bytes(b)
    palette = cluster_palette(im, k=3)
    lighting, camera = guess_lighting_and_camera(im)
    labels = google_vision_labels(b)

    subject = None
    environment = None
    for l in labels:
        s = l.lower()
        if not subject and any(k in s for k in ["person","human","man","woman","girl","boy","dog","cat","bird"]):
            subject = l
        if not environment and any(k in s for k in ["street","city","bridge","forest","beach","mountain","building","harbor"]):
            environment = l

    return Analysis(
        exif=exif,
        technical=Technical(lighting=lighting, camera=camera, palette_60_30_10=palette, atmosphere=[]),
        narrative=Narrative(subject=subject or "unknown subject",
                            emotion=None,
                            environment=environment or "unknown environment",
                            action_main=None,
                            secondary_characters=[],
                            props=[],
                            interaction=None),
        confidence={"exif": 0.95 if exif.camera_model or exif.focal_length_mm else 0.5,
                    "vision": 0.9 if labels else 0.5}
    )
