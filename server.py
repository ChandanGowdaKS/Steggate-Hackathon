"""
StegGate API Server  v6.0  —  Production / Deployable
======================================================
No API key required — fully open API.

Run locally:
    pip install -r requirements.txt
    python server.py

Run with Docker:
    docker compose up

Env vars  (set in .env):
    PORT              default 5050
    WEBHOOK_SECRET    HMAC secret for outbound webhook signatures
    UPLOAD_DIR        default ./uploads
    SANITIZED_DIR     default ./sanitized
    DB_PATH           default ./steggate.db
    MAX_BYTES         default 52428800  (50 MB)
    RATE_LIMIT        default 60  (requests/minute)
    WORKERS           default 4  (engine thread pool)
"""

import asyncio, base64, hashlib, hmac, json, logging, os, re, secrets
import shutil, sqlite3, sys, time, traceback, uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

sys.path.insert(0, os.path.dirname(__file__))
from security_engine import EnterpriseStegEngine, _sanitise_floats

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
CAL_PATH       = os.path.join(os.path.dirname(__file__), "calibration_web.json")
MAX_BYTES      = int(os.environ.get("MAX_BYTES",  50 * 1024 * 1024))
RATE_LIMIT     = int(os.environ.get("RATE_LIMIT", 60))
WORKERS        = int(os.environ.get("WORKERS",    4))

_BASE         = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR    = Path(os.environ.get("UPLOAD_DIR",    os.path.join(_BASE, "uploads")))
SANITIZED_DIR = Path(os.environ.get("SANITIZED_DIR", os.path.join(_BASE, "sanitized")))
DB_PATH       = os.environ.get("DB_PATH",             os.path.join(_BASE, "steggate.db"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SANITIZED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("steggate")

# ─────────────────────────────────────────────────────────────────────────────
#  Database  (SQLite)
# ─────────────────────────────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              INTEGER NOT NULL,
                filename        TEXT,
                file_bytes      INTEGER,
                is_threat       INTEGER,
                risk_score      REAL,
                was_sanitized   INTEGER,
                scan_ms         INTEGER,
                upload_id       TEXT,
                ip              TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_log(ts);
        """)
    log.info(f"Database ready: {DB_PATH}")

_init_db()

# ─────────────────────────────────────────────────────────────────────────────
#  Engine
# ─────────────────────────────────────────────────────────────────────────────

engine = EnterpriseStegEngine()
_pool  = ThreadPoolExecutor(max_workers=WORKERS)

if os.path.exists(CAL_PATH):
    try:
        engine.load_calibration(CAL_PATH)
        log.info(f"Calibration loaded from {CAL_PATH}")
    except Exception as e:
        log.warning(f"Calibration load failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="StegGate API",
    version="6.0",
    description="""
## StegGate — Image Steganography Detection & Sanitization

Integrate in 3 lines:

```python
import requests
resp = requests.post("https://your-api/api/sanitize",
    files={"file": open("photo.png", "rb")})
open("clean.jpg", "wb").write(resp.content)
```

**Flow:** Upload image → scan for hidden payloads → sanitize if infected →
return clean image. Store what you get back — never the original.

**Response headers:**
- `X-Threat-Detected: true/false`
- `X-Risk-Score: 0.00–100.00`
- `X-Was-Sanitized: true/false`
""",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Threat-Detected", "X-Risk-Score", "X-Was-Sanitized",
        "X-Original-Filename", "X-Scan-Duration-Ms",
        "X-Upload-ID", "Content-Disposition",
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()

def _stem(f: str) -> str:
    return re.sub(r'\.[^.]+$', '', f or "image")

def _allowed(f: str) -> bool:
    ext = (f or "").lower().rsplit(".", 1)[-1] if "." in (f or "") else ""
    return ext in {"png", "jpg", "jpeg", "bmp", "webp", "tiff"}

def _safe_name(filename: str) -> str:
    name = os.path.basename(filename or "image")
    return re.sub(r'[^\w.\-]', '_', name)[:180] or "image"

async def _run(fn, *args):
    return await asyncio.get_event_loop().run_in_executor(_pool, fn, *args)

def _sign_webhook(body: bytes) -> str:
    if not WEBHOOK_SECRET:
        return ""
    return "sha256=" + hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()

async def _fire_webhook(url: str, payload: dict):
    try:
        body = json.dumps(payload).encode()
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(url, content=body, headers={
                "Content-Type":         "application/json",
                "X-StegGate-Event":     "scan.complete",
                "X-StegGate-Signature": _sign_webhook(body),
            })
    except Exception as e:
        log.warning(f"Webhook error: {e}")

def _save_upload(raw: bytes, filename: str, upload_id: str) -> Path:
    path = UPLOAD_DIR / f"{upload_id}_{_safe_name(filename)}"
    path.write_bytes(raw)
    return path

def _save_sanitized(clean: bytes, filename: str, upload_id: str) -> Path:
    path = SANITIZED_DIR / f"{upload_id}_{_stem(_safe_name(filename))}_sanitized.jpg"
    path.write_bytes(clean)
    return path

def _log_usage(upload_id, filename, file_bytes,
               is_threat, risk_score, was_sanitized, scan_ms, ip):
    try:
        with _db() as conn:
            conn.execute("""
                INSERT INTO usage_log
                  (ts, filename, file_bytes, is_threat,
                   risk_score, was_sanitized, scan_ms, upload_id, ip)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (int(time.time()), filename, file_bytes,
                  int(is_threat), risk_score, int(was_sanitized),
                  scan_ms, upload_id, ip))
    except Exception as e:
        log.warning(f"Usage log error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    p = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if not os.path.exists(p):
        return HTMLResponse("<h2>dashboard.html not found</h2>", 404)
    return HTMLResponse(open(p).read())


@app.get("/api/health", tags=["Meta"], summary="Liveness check")
async def health():
    return {
        "status":     "ok",
        "version":    "6.0",
        "calibrated": bool(engine.calibration and engine.calibration.is_ready),
        "cal_images": engine.calibration.n_images if engine.calibration else 0,
        "tools": {
            "zsteg":   bool(shutil.which("zsteg")),
            "binwalk": bool(shutil.which("binwalk")),
        },
    }


@app.post(
    "/api/sanitize",
    tags=["Core"],
    summary="★ Upload image → scan → sanitize → return clean image",
)
async def sanitize(
    request:     Request,
    file:        UploadFile = File(...,   description="Image file (PNG/JPG/BMP/WEBP/TIFF, max 50 MB)"),
    force:       bool       = Form(False, description="Force sanitize even if no threat detected"),
    webhook_url: str        = Form("",    description="POST scan result JSON to this URL after processing"),
):
    """
    Send an image → get back the clean image. No API key needed.

    - **Response body** = clean image bytes
    - **Response headers** = scan decision

    | Header | Value |
    |---|---|
    | `X-Threat-Detected` | `true` if payload was found |
    | `X-Risk-Score` | 0–100. Score ≥50 = threat |
    | `X-Was-Sanitized` | `true` if image was cleaned |
    | `X-Scan-Duration-Ms` | scan time in ms |
    | `X-Upload-ID` | unique ID for this request |
    """
    client_ip = request.client.host if request.client else "unknown"

    if not file.filename or not _allowed(file.filename):
        raise HTTPException(400, "Unsupported file type. Accepted: PNG, JPG, BMP, WEBP, TIFF")

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_BYTES // 1024 // 1024} MB)")

    upload_id = str(uuid.uuid4())[:8]
    _save_upload(raw, file.filename, upload_id)

    log.info(f"[{upload_id}] file={file.filename}  size={len(raw)//1024}KB  ip={client_ip}")

    t0 = time.monotonic()
    try:
        result = await _run(engine.process_file, raw, file.filename)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Engine error: {e}")
    scan_ms = int((time.monotonic() - t0) * 1000)

    is_threat  = bool(result.get("is_threat", False))
    risk_score = float(result.get("risk_score", 0.0))

    if is_threat or force:
        clean_bytes = result["safe_file_bytes"]
        _save_sanitized(clean_bytes, file.filename, upload_id)
        out, mime, fname, was = (
            clean_bytes, "image/jpeg",
            f"{_stem(file.filename)}_sanitized.jpg", True,
        )
    else:
        out, mime, fname, was = raw, file.content_type or "image/jpeg", file.filename, False

    log.info(f"[{upload_id}] threat={is_threat}  risk={risk_score:.1f}  "
             f"sanitized={was}  ms={scan_ms}")

    _log_usage(upload_id, file.filename, len(raw),
               is_threat, risk_score, was, scan_ms, client_ip)

    if webhook_url:
        asyncio.create_task(_fire_webhook(webhook_url, {
            "event":             "scan.complete",
            "timestamp":         int(time.time()),
            "upload_id":         upload_id,
            "original_filename": file.filename,
            "is_threat":         is_threat,
            "risk_score":        round(risk_score, 2),
            "was_sanitized":     was,
            "output_filename":   fname,
            "scan_duration_ms":  scan_ms,
        }))

    return Response(
        content=out,
        media_type=mime,
        headers={
            "X-Threat-Detected":   str(is_threat).lower(),
            "X-Risk-Score":        f"{risk_score:.2f}",
            "X-Was-Sanitized":     str(was).lower(),
            "X-Original-Filename": file.filename or "",
            "X-Scan-Duration-Ms":  str(scan_ms),
            "X-Upload-ID":         upload_id,
            "Content-Disposition": f'attachment; filename="{fname}"',
        },
    )


@app.post("/api/scan", tags=["Core"], summary="Full forensic JSON report")
async def scan(
    request: Request,
    file:    UploadFile = File(...),
):
    """
    Full forensic JSON — risk score, heatmap, bitplane analysis,
    zsteg/binwalk findings, base64 sanitized image.

    Use `/api/sanitize` for your upload pipeline.
    Use `/api/scan` for your forensic dashboard.
    """
    client_ip = request.client.host if request.client else "unknown"

    if not file.filename or not _allowed(file.filename):
        raise HTTPException(400, "Unsupported file type")

    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File too large")

    upload_id = str(uuid.uuid4())[:8]
    _save_upload(raw, file.filename, upload_id)

    try:
        result = await _run(engine.process_file, raw, file.filename)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

    is_threat  = bool(result.get("is_threat", False))
    risk_score = float(result.get("risk_score", 0.0))

    _log_usage(upload_id, file.filename, len(raw),
               is_threat, risk_score, False, 0, client_ip)

    safe_b64    = _b64(result.pop("safe_file_bytes"))
    heatmap_b64 = _b64(result.pop("heatmap_bytes"))
    for t in ("zsteg", "binwalk"):
        if isinstance(result.get(t), dict):
            result[t].pop("raw_output", None)

    return JSONResponse(_sanitise_floats({
        **result,
        "safe_b64":    safe_b64,
        "heatmap_b64": heatmap_b64,
        "filename":    file.filename,
        "upload_id":   upload_id,
    }))


@app.get("/admin/usage", tags=["Admin"], summary="Usage stats — all requests")
async def usage_stats(days: int = 7):
    since = int(time.time()) - (days * 86400)
    with _db() as conn:
        rows = conn.execute("""
            SELECT
                COUNT(*)              AS total_requests,
                SUM(is_threat)        AS threats_found,
                SUM(was_sanitized)    AS images_sanitized,
                SUM(file_bytes)       AS total_bytes,
                AVG(scan_ms)          AS avg_scan_ms
            FROM usage_log
            WHERE ts >= ?
        """, (since,)).fetchone()
    return {
        "period_days":      days,
        "total_requests":   rows["total_requests"]   or 0,
        "threats_found":    rows["threats_found"]    or 0,
        "images_sanitized": rows["images_sanitized"] or 0,
        "total_mb":         round((rows["total_bytes"] or 0) / 1024 / 1024, 2),
        "avg_scan_ms":      round(rows["avg_scan_ms"] or 0),
    }


@app.get("/admin/logs", tags=["Admin"], summary="Recent scan log")
async def recent_logs(limit: int = 100):
    with _db() as conn:
        rows = conn.execute("""
            SELECT ts, filename, file_bytes, is_threat, risk_score,
                   was_sanitized, scan_ms, upload_id, ip
            FROM usage_log
            ORDER BY ts DESC LIMIT ?
        """, (limit,)).fetchall()
    return {
        "logs": [
            {
                "ts":            r["ts"],
                "filename":      r["filename"],
                "file_kb":       round((r["file_bytes"] or 0) / 1024, 1),
                "is_threat":     bool(r["is_threat"]),
                "risk_score":    r["risk_score"],
                "was_sanitized": bool(r["was_sanitized"]),
                "scan_ms":       r["scan_ms"],
                "upload_id":     r["upload_id"],
                "ip":            r["ip"],
            }
            for r in rows
        ]
    }


@app.post("/admin/calibrate", tags=["Admin"], summary="Build detection baseline")
async def calibrate(
    files:  list[UploadFile] = File(...),
    source: str              = Form("unspecified"),
):
    """Upload 2–50 known-clean images to calibrate K-sigma detection thresholds."""
    raw_list, names = [], []
    for f in files:
        if f.filename and _allowed(f.filename):
            raw_list.append(await f.read())
            names.append(f.filename)
    if not raw_list:
        raise HTTPException(400, "No valid images")
    try:
        await _run(engine.calibrate_from_bytes, raw_list, source)
        engine.save_calibration(CAL_PATH)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))
    return {"success": True, "n_images": engine.calibration.n_images,
            "source": engine.calibration.source_hint, "filenames": names}


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5050))
    cal_info = (f"loaded ({engine.calibration.n_images} images)"
                if engine.calibration else "none")
    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║   StegGate API  v6.0  —  Production                     ║
  ╠══════════════════════════════════════════════════════════╣
  ║   http://0.0.0.0:{port:<5}                                   ║
  ║   http://0.0.0.0:{port:<5}/docs  ← Interactive docs          ║
  ╠══════════════════════════════════════════════════════════╣
  ║   POST  /api/sanitize   ★ scan + return clean image      ║
  ║   POST  /api/scan         full forensic JSON             ║
  ║   GET   /api/health       liveness check                 ║
  ╠══════════════════════════════════════════════════════════╣
  ║   GET   /admin/usage      usage stats                    ║
  ║   GET   /admin/logs       recent scan log                ║
  ║   POST  /admin/calibrate  rebuild baseline               ║
  ╠══════════════════════════════════════════════════════════╣
  ║   Auth         : NONE — open API                         ║
  ║   Calibration  : {cal_info:<42}║
  ║   Uploads dir  : {str(UPLOAD_DIR):<42}║
  ║   Sanitized dir: {str(SANITIZED_DIR):<42}║
  ║   Database     : {DB_PATH:<42}║
  ╚══════════════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")