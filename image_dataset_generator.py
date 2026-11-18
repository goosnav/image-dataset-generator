#!/usr/bin/env python3
"""Comic image -> spatially faithful, ordered play scripts. Python 3.10+."""
from __future__ import annotations

import argparse
import base64
import csv
import getpass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError:
    print("Pillow is required. Install with: python -m pip install Pillow", file=sys.stderr)
    raise SystemExit(2)

VERSION = "0.1.0"
API_ROOT = "https://api.openai.com/v1"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
# USD per million tokens, standard non-regional pricing, checked November 2026.
# Batch: 50% of standard. Any changes in model rates require updating this table.
PRICES = {
    "gpt-6.2-sol": (2.0, 10.0, 0.10),
    "gpt-6.2-astra": (8.0, 40.0, 0.40),
    "gpt-6.1-sol": (2.0, 10.0, 0.10),
    "gpt-6.1-terra": (1.0, 5.0, 0.05),
    "gpt-6-astra": (5.0, 25.0, 0.50),
    "gpt-5.4-mini": (0.75, 4.50, 0.075),
}
DEFAULT_MODEL = "gpt-6.2-sol"
FINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}
SYSTEM_PROMPT = """You are a meticulous comic archivist, production storyboard artist and transcript editor.
Analyze the ACTUAL supplied image. Return strictly the requested JSON schema.
Objective: a third party should be able to redraw the source with high fidelity using the script alone.
1. Resolve panel boundaries and WESTERN reading order (top to bottom, left to right, respecting staggered gutters, inset panels and intentional exceptions). Give each panel normalized bounding coordinates [x,y,width,height], values 0..1000 relative to the entire image. Include unbordered panels when clearly implied. Do not treat speech bubbles as panels.
2. Transcribe EVERY visible dialogue balloon, narration, sign, label, effect and relevant background text VERBATIM (case, spelling, punctuation, line breaks when visible). Attribute speaker only when supported by tails or context. Preserve the order of balloons within the panel. For illegible words use [illegible] or [unclear: best guess] and log uncertainty. Never supply plausible dialogue instead of observed text.
3. In each panel describe staging/blocking with precise left/right/center placement, depth, shot size, viewpoint, facing of torso and head, eye gaze, gestures, limb positions, contact points, expressions, props, and spatial relationships. Separate anatomical left/right from image-left/image-right. Describe visual changes panel to panel, but do NOT invent off-screen events or character intent.
4. Record actual line art and styling: stroke thickness and quality, hatching, uneven ink edges, silhouette, proportions, lettering, borders and gutters, colors or lack thereof, negative space, backgrounds and textures. Do not insert generic studio/comic-style assumptions; the visual image is the source of truth.
5. Each character's visible design must be repeatably drawable: head shape, face, hair, clothing, accessories, distinctive marks, scale. Record continuity. Reference characters by stable IDs across panels; no fabricated names.
6. Describe concrete visible details before interpretation. Explicitly distinguish ambiguity and guesses. Do not moralize, summarize away details, over-explain the joke or invent lore. Empty lists are acceptable where none exist.
7. Keep the output detailed but information-dense: describe what a careful artist NEEDS to see. Within each panel, provide accurate precise redraw directions; do not rely on previous panels to describe changes. If a drawing contains a single illustration rather than a comic, record it as one panel.
"""

def obj(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": list(properties)}

def arr(items: dict) -> dict:
    return {"type": "array", "items": items}

S = {"type": "string"}
N = {"type": "integer"}
BOX = arr(N)
SCHEMA = obj({
    "title_or_visible_heading": S,
    "overview": S,
    "reading_order_notes": S,
    "whole_page_style": S,
    "layout_and_gutters": S,
    "cast": arr(obj({"id": S, "design": S, "continuity_notes": S})),
    "panels": arr(obj({
        "number": N,
        "bounds_0_to_1000_xywh": BOX,
        "shot_and_viewpoint": S,
        "setting_and_background": S,
        "composition_and_depth": S,
        "actors": arr(obj({
            "id": S, "position_and_scale": S, "torso_facing": S,
            "head_facing_and_gaze": S, "pose_and_hands": S,
            "expression": S, "visible_design": S, "interaction": S
        })),
        "props": arr(obj({"name": S, "position": S, "appearance_and_contact": S})),
        "text": arr(obj({
            "order": N, "kind": S, "speaker_id_or_unknown": S,
            "verbatim": S, "placement_and_balloon_tail": S,
            "lettering_appearance": S, "legibility": S
        })),
        "visible_action_and_transition": S,
        "exact_redraw_directions": S,
        "uncertainties": arr(S)
    })),
    "ending_or_final_beat": S,
    "redraw_checklist": arr(S),
    "global_uncertainties": arr(S)
})


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "image-dataset-generator" / "config.json"


def read_config() -> dict:
    try:
        return json.loads(config_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def set_key() -> None:
    key = getpass.getpass("OpenAI API key (hidden): ").strip()
    if not key:
        raise ValueError("Empty API key: no changes made")
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = read_config()
    cfg["api_key"] = key
    # Atomic, owner-readable on Unix. Windows: normal user-profile ACLs apply.
    temp = path.with_suffix(".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(temp, path)
    if os.name != "nt":
        path.chmod(0o600)
    print(f"API key stored in your user profile: {path} (never exported with datasets)")


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY") or read_config().get("api_key")
    if not key:
        raise RuntimeError("No API key. Run 'image-dataset-generator configure' or set OPENAI_API_KEY")
    return key


class APIError(RuntimeError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"HTTP {status}: {message[:700]}")


class API:
    """Small REST client. No SDK dependency; no automatic replay of uncertain charges."""
    def __init__(self, key: str):
        self.key = key

    def _request(self, method: str, path: str, payload=None, content_type="application/json", timeout=240):
        body = (json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if isinstance(payload, (dict, list)) else payload)
        headers = {"Authorization": f"Bearer {self.key}", "User-Agent": f"comic-dataset/{VERSION}"}
        if body is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(API_ROOT + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                if path.endswith("/content"):
                    return raw
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise APIError(exc.code, exc.read().decode("utf-8", "replace")) from exc

    def response(self, body: dict) -> dict:
        return self._request("POST", "/responses", body, timeout=300)

    def upload(self, data: bytes) -> str:
        boundary = "comicdataset" + uuid.uuid4().hex
        head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"requests.jsonl\"\r\n"
                f"Content-Type: application/jsonl\r\n\r\n").encode()
        raw = head + data + f"\r\n--{boundary}--\r\n".encode()
        res = self._request("POST", "/files", raw,
                            f"multipart/form-data; boundary={boundary}", timeout=300)
        return res["id"]

    def create_batch(self, file_id: str) -> dict:
        return self._request("POST", "/batches", {"input_file_id": file_id,
                "endpoint": "/v1/responses", "completion_window": "24h"})

    def batch(self, batch_id: str) -> dict:
        return self._request("GET", "/batches/" + batch_id)

    def batch_content(self, file_id: str) -> bytes:
        return self._request("GET", f"/files/{file_id}/content")


def scan_images(root: Path, recursive: bool, output: Path | None = None) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    candidates = root.rglob("*") if recursive else root.iterdir()
    selected = []
    for path in candidates:
        if output and (path == output or output in path.parents):
            continue
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            # Avoid any old exports nested inside the image folder.
            if any(p.startswith("comic-dataset-export-") for p in path.relative_to(root).parts[:-1]):
                continue
            selected.append(path)
    return sorted(selected, key=lambda p: p.relative_to(root).as_posix().casefold())


def image_data(path: Path, max_side: int) -> tuple[str, int, int, int, int]:
    """Return data URL, original dimensions, and sent dimensions. Original file is never changed."""
    Image.MAX_IMAGE_PIXELS = 120_000_000
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source)
        orig_w, orig_h = im.size
        if im.mode not in ("RGB", "L"):
            if "A" in im.getbands():
                rgba = im.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, "white")
                bg.alpha_composite(rgba)
                im = bg.convert("RGB")
            else:
                im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        width, height = im.size
        # Lossless PNG protects small handwritten glyphs and fine ink strokes.
        buffer = io.BytesIO()
        im.save(buffer, "PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded, orig_w, orig_h, width, height


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(buf)
    return sha.hexdigest()


def build_body(model: str, data_uri: str, filename: str, original: tuple[int, int],
               max_tokens: int, reasoning: str, focus: str = "") -> dict:
    text = (f"File: {filename}. Source pixel dimensions: {original[0]}x{original[1]}. "
            "Describe only what is actually visible; return complete JSON. " + focus)
    return {"model": model,
            "input": [{"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                      {"role": "user", "content": [{"type": "input_text", "text": text},
                          {"type": "input_image", "image_url": data_uri, "detail": "original"}]}],
            "reasoning": {"effort": reasoning},
            "text": {"format": {"type": "json_schema", "name": "comic_transcript_v1",
                                "strict": True, "schema": SCHEMA}, "verbosity": "high"},
            "max_output_tokens": max_tokens, "store": False}


def price(model: str, batch: bool) -> tuple[float, float, float]:
    if model not in PRICES:
        raise ValueError(f"Unknown model {model}. Supported price cards: {', '.join(PRICES)}")
    multiplier = 0.5 if batch else 1.0
    return tuple(v * multiplier for v in PRICES[model])


def reserve_cost(model: str, batch: bool, width: int, height: int,
                 max_tokens: int, user_text: str = "") -> float:
    """Conservative per-request reservation, released when actual usage is known.

    Max output is an API-enforced upper bound, input is an estimate with margin.
    This is a per-run cost safeguard, not an account-level billing limit.
    """
    input_rate, output_rate, _ = price(model, batch)
    pixels = math.ceil(width / 32) * math.ceil(height / 32)
    image_tokens = math.ceil(pixels * 1.2)
    # More than 2 characters/token allowance for mostly ASCII prompts.
    prompt_chars = len(SYSTEM_PROMPT) + len(user_text) + 1500
    text_tokens = math.ceil(prompt_chars / 2) + 500
    return round((input_rate * (image_tokens + text_tokens) * 1.25 +
                  output_rate * max_tokens) / 1_000_000, 8)


def actual_cost(model: str, batch: bool, usage: dict | None, fallback: float) -> float:
    if not usage:
        return fallback
    inp = usage.get("input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
    normal, output, cached_rate = price(model, batch)
    return round(((inp - min(inp, cached)) * normal + min(inp, cached) * cached_rate +
                  out * output) / 1_000_000, 8)


def parse_response(body: dict) -> dict:
    if body.get("status") != "completed":
        raise ValueError(f"Response not completed: {body.get('status')} / {body.get('incomplete_details')}")
    texts = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "refusal":
                    raise ValueError("Model refusal: " + content.get("refusal", ""))
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))
    if not texts:
        raise ValueError("No text in completed response")
    data = json.loads("".join(texts))
    if not isinstance(data, dict) or not data.get("panels"):
        raise ValueError("No panels in response")
    for i, panel in enumerate(data["panels"], 1):
        if panel.get("number") != i:
            raise ValueError(f"Panel {i} numbered incorrectly")
        box = panel.get("bounds_0_to_1000_xywh", [])
        if len(box) != 4 or any(not isinstance(v, int) or not 0 <= v <= 1000 for v in box):
            raise ValueError(f"Panel {i} has invalid bounding box")
        if box[0] + box[2] > 1010 or box[1] + box[3] > 1010:
            raise ValueError(f"Panel {i} extends beyond image")
    return data


def abridge(detailed: dict) -> dict:
    panels = []
    for p in detailed["panels"]:
        acts = p.get("actors", [])
        stage = "; ".join(f"{a['id']}: {a['position_and_scale']}; {a['torso_facing']}; "
                          f"{a['pose_and_hands']}; {a['expression']}" for a in acts)
        dialogue = [{"speaker": t["speaker_id_or_unknown"], "kind": t["kind"],
                     "text": t["verbatim"]} for t in p.get("text", [])]
        panels.append({"panel": p["number"], "shot": p["shot_and_viewpoint"],
                       "action": p["visible_action_and_transition"],
                       "blocking": stage, "dialogue": dialogue})
    return {"title": detailed.get("title_or_visible_heading", ""),
            "overview": detailed.get("overview", ""),
            "style": detailed.get("whole_page_style", ""),
            "panels": panels,
            "ending": detailed.get("ending_or_final_beat", "")}


def review_reasons(detailed: dict) -> list[str]:
    reasons = list(detailed.get("global_uncertainties", []))
    for p in detailed.get("panels", []):
        reasons += [f"Panel {p['number']}: {v}" for v in p.get("uncertainties", [])]
        for t in p.get("text", []):
            legibility = t.get("legibility", "").lower()
            if any(term in legibility for term in ("unclear", "illegible", "low", "uncertain")) or "[illegible]" in t.get("verbatim", ""):
                reasons.append(f"Panel {p['number']}: low-confidence text: {t.get('verbatim', '')[:100]}")
    return reasons


def connect(export: Path) -> sqlite3.Connection:
    db = sqlite3.connect(export / "dataset.sqlite3")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    return db


def init_db(db: sqlite3.Connection) -> None:
    db.executescript("""
    CREATE TABLE IF NOT EXISTS images (
      id TEXT PRIMARY KEY, relative_path TEXT NOT NULL UNIQUE,
      sha256 TEXT NOT NULL, width INTEGER, height INTEGER,
      status TEXT NOT NULL DEFAULT 'pending', phase TEXT NOT NULL DEFAULT 'base',
      batch_id TEXT, reserved_usd REAL NOT NULL DEFAULT 0,
      cost_usd REAL NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0,
      output_tokens INTEGER NOT NULL DEFAULT 0, cached_tokens INTEGER NOT NULL DEFAULT 0,
      model TEXT, attempts INTEGER NOT NULL DEFAULT 0,
      detailed TEXT, abridged TEXT, error TEXT, updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS batches (
      id TEXT PRIMARY KEY, input_file_id TEXT NOT NULL, status TEXT NOT NULL,
      collected INTEGER NOT NULL DEFAULT 0, submitted_at TEXT NOT NULL,
      request_count INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_status ON images(status);
    CREATE INDEX IF NOT EXISTS idx_batch_id ON images(batch_id);
    """)
    db.commit()


def load_manifest(export: Path) -> dict:
    return json.loads((export / "manifest.json").read_text(encoding="utf-8"))


def save_manifest(export: Path, data: dict) -> None:
    temp = export / "manifest.json.tmp"
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(export / "manifest.json")


def create_export(root: Path, opts: dict, output_base: Path | None = None) -> Path:
    base = (output_base or Path.cwd()).resolve()
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / ("comic-dataset-export-" + timestamp())
    count = 1
    while candidate.exists():
        candidate = base / ("comic-dataset-export-" + timestamp() + f"-{count:02}")
        count += 1
    candidate.mkdir()
    manifest = {"generator": VERSION, "created_at": now(), "source_folder": str(root.resolve()),
                "options": opts, "pricing_usd_per_million": {k: list(v) for k, v in PRICES.items()},
                "warning": "Estimates are safeguards, not an OpenAI account spending limit. Prices must be verified."}
    save_manifest(candidate, manifest)
    db = connect(candidate)
    init_db(db)
    db.close()
    return candidate


def add_sources(export: Path, root: Path, recursive: bool) -> int:
    files = scan_images(root, recursive, export)
    db = connect(export)
    with db:
        for file in files:
            rel = file.relative_to(root).as_posix()
            identity = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:20]
            digest = sha256_file(file)
            existing = db.execute("SELECT sha256 FROM images WHERE id=?", (identity,)).fetchone()
            if existing and existing["sha256"] != digest:
                raise ValueError(f"Source changed since last run: {rel}. Export is immutable. Start a new run.")
            if not existing:
                db.execute("INSERT INTO images(id,relative_path,sha256,updated_at) VALUES(?,?,?,?)",
                           (identity, rel, digest, now()))
    db.close()
    return len(files)


def spend(db: sqlite3.Connection) -> tuple[float, float]:
    row = db.execute("SELECT COALESCE(SUM(cost_usd),0) AS spent, "
                     "COALESCE(SUM(reserved_usd),0) AS reserved FROM images").fetchone()
    return float(row["spent"]), float(row["reserved"])


def mark_result(db: sqlite3.Connection, row: sqlite3.Row, response: dict,
                model: str, is_batch: bool) -> None:
    usage = response.get("usage") or {}
    billed = actual_cost(model, is_batch, usage, float(row["reserved_usd"]))
    inp = int(usage.get("input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)
    cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0)
    previous_cost = float(row["cost_usd"])
    try:
        data = parse_response(response)
        brief = abridge(data)
        db.execute("""UPDATE images SET status='done', phase=?, detailed=?, abridged=?, error=NULL,
            model=?, cost_usd=?, reserved_usd=0, batch_id=NULL,
            input_tokens=input_tokens+?, output_tokens=output_tokens+?,
            cached_tokens=cached_tokens+?, updated_at=? WHERE id=?""",
            (row["phase"], json.dumps(data, ensure_ascii=False), json.dumps(brief, ensure_ascii=False),
             model, previous_cost + billed, inp, out, cached, now(), row["id"]))
    except (ValueError, KeyError, TypeError) as exc:
        state = "done" if row["detailed"] else "error"
        db.execute("""UPDATE images SET status=?, error=?, cost_usd=?, reserved_usd=0,
            batch_id=NULL, input_tokens=input_tokens+?, output_tokens=output_tokens+?,
            cached_tokens=cached_tokens+?, updated_at=? WHERE id=?""",
            (state, f"Invalid/partial model response: {exc}", previous_cost + billed,
             inp, out, cached, now(), row["id"]))


def mark_error(db: sqlite3.Connection, row: sqlite3.Row, message: str) -> None:
    state = "done" if row["detailed"] else "error"
    # An HTTP-level failure with unknown usage reserves the full per-request amount.
    reserve = float(row["reserved_usd"])
    db.execute("""UPDATE images SET status=?, error=?, cost_usd=cost_usd+?,
        reserved_usd=0, batch_id=NULL, updated_at=? WHERE id=?""",
        (state, message[:1200], reserve, now(), row["id"]))


def candidates(db: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(db.execute("SELECT * FROM images WHERE status='pending' ORDER BY relative_path"))


def request_for(db: sqlite3.Connection, row: sqlite3.Row, root: Path,
                opts: dict, model: str, batch: bool) -> tuple[dict, float, int, int]:
    path = root / Path(row["relative_path"])
    if not path.is_file() or sha256_file(path) != row["sha256"]:
        raise ValueError("Source missing or modified since scan: " + row["relative_path"])
    data_uri, w, h, sent_w, sent_h = image_data(path, opts["max_side"])
    focus = ""
    if row["phase"] == "refine" and row["detailed"]:
        reasons = review_reasons(json.loads(row["detailed"]))
        focus = "REVIEW PASS: scrutinize ambiguous text, tiny visual details and panel boundaries. "
        if reasons:
            focus += "Earlier transcription concerns (verify against image, not as facts): " + "; ".join(reasons)[:800]
    body = build_body(model, data_uri, row["relative_path"], (w, h),
                      opts["max_output_tokens"], opts["reasoning"], focus)
    # URL base64 is image data, not free-form prompt text; calculate image separately.
    reservation = reserve_cost(model, batch, sent_w, sent_h,
                               opts["max_output_tokens"], row["relative_path"] + focus)
    return body, reservation, w, h


def run_live(export: Path, api: API, limit: int = 0) -> None:
    manifest = load_manifest(export)
    opts = manifest["options"]
    model = opts["model"]
    db = connect(export)
    root = Path(manifest["source_folder"])
    completed = 0
    for row in candidates(db):
        if limit and completed >= limit:
            break
        try:
            body, reserve, w, h = request_for(db, row, root, opts, model, False)
            spent, pending = spend(db)
            if spent + pending + reserve > opts["hard_cap_usd"] - 0.1:
                print(f"Stopped safely at cap; {row['relative_path']} requires ${reserve:.3f} reserved")
                break
            with db:
                db.execute("""UPDATE images SET status='submitted', batch_id='live', reserved_usd=?,
                    width=?, height=?, model=?, attempts=attempts+1, updated_at=? WHERE id=?""",
                    (reserve, w, h, model, now(), row["id"]))
            current = db.execute("SELECT * FROM images WHERE id=?", (row["id"],)).fetchone()
            try:
                response = api.response(body)
                with db:
                    mark_result(db, current, response, model, False)
            except (APIError, urllib.error.URLError, TimeoutError) as exc:
                # No retries: after a timeout, the server MAY have billed a response.
                # Keep the full reservation and explicit error for reconciliation.
                with db:
                    mark_error(db, current, f"API call failed (may have billed): {exc}")
                print(f"ERROR {row['relative_path']}: {exc}")
                if not isinstance(exc, APIError) or exc.status in (401, 402, 403, 429):
                    break
            completed += 1
            spent, _ = spend(db)
            print(f"[{completed}] {row['relative_path']} | reserved ${reserve:.3f}, tracked ${spent:.2f}")
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            with db:
                current = db.execute("SELECT * FROM images WHERE id=?", (row["id"],)).fetchone()
                if current["status"] == "submitted":
                    mark_error(db, current, f"Uncertain live response: {exc}")
                else:
                    db.execute("UPDATE images SET status='error', error=?, updated_at=? WHERE id=?",
                               (str(exc)[:1200], now(), row["id"]))
            print(f"INVALID {row['relative_path']}: {exc}")
    db.close()
    export_files(export)


def submit_batches(export: Path, api: API, limit: int = 0,
                   max_file_mb: int = 24, batch_items: int = 45) -> int:
    manifest = load_manifest(export)
    opts = manifest["options"]
    db = connect(export)
    root = Path(manifest["source_folder"])
    model = opts["model"]
    remaining = candidates(db)
    if limit:
        remaining = remaining[:limit]
    submitted = 0
    while remaining:
        spent, held = spend(db)
        lines = []
        chosen = []
        payload_size = 0
        for row in remaining[:batch_items]:
            try:
                body, reserve, w, h = request_for(db, row, root, opts, model, True)
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                with db:
                    db.execute("UPDATE images SET status='error', error=?, updated_at=? WHERE id=?",
                               (str(exc)[:1200], now(), row["id"]))
                print(f"INVALID {row['relative_path']}: {exc}")
                continue
            if spent + held + sum(x[2] for x in chosen) + reserve > opts["hard_cap_usd"] - 0.10:
                break
            record = {"custom_id": row["id"], "method": "POST", "url": "/v1/responses", "body": body}
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
            if len(line) > max_file_mb * 1024 * 1024:
                with db:
                    db.execute("UPDATE images SET status='error', error='Single prepared image exceeds batch file limit' WHERE id=?", (row["id"],))
                continue
            if payload_size + len(line) > max_file_mb * 1024 * 1024:
                break
            lines.append(line)
            chosen.append((row, (w, h), reserve))
            payload_size += len(line)
        if not chosen:
            break
        # Once submitted, never retry automatically; request mapping persists locally.
        file_id = api.upload(b"".join(lines))
        # Mark uncertain before the non-idempotent creation call. Do not silently
        # duplicate a possibly accepted batch after a network timeout/crash.
        with db:
            for row, (w, h), reserve in chosen:
                db.execute("""UPDATE images SET status='sending', batch_id=?, reserved_usd=?,
                    width=?, height=?, model=?, attempts=attempts+1, updated_at=? WHERE id=?""",
                    ("unconfirmed:" + file_id, reserve, w, h, model, now(), row["id"]))
        try:
            batch = api.create_batch(file_id)
        except Exception:
            print(f"UNCERTAIN submission for file {file_id}. DO NOT rerun these items. "
                  f"Check OpenAI Batches, then attach-batch {export} <batch_id>.", file=sys.stderr)
            raise
        with db:
            db.execute("INSERT INTO batches(id,input_file_id,status,submitted_at,request_count) VALUES(?,?,?,?,?)",
                       (batch["id"], file_id, batch["status"], now(), len(chosen)))
            for row, _, _ in chosen:
                db.execute("UPDATE images SET status='submitted', batch_id=?, updated_at=? WHERE id=?",
                           (batch["id"], now(), row["id"]))
        submitted += len(chosen)
        print(f"Batch {batch['id']}: {len(chosen)} images, ${sum(x[2] for x in chosen):.2f} reserved, {payload_size/1e6:.1f} MB")
        remaining = candidates(db)
        if limit:
            if submitted >= limit:
                break
            remaining = remaining[:max(0, limit - submitted)]
    db.close()
    return submitted


def collect_batches(export: Path, api: API) -> tuple[int, int]:
    db = connect(export)
    finished = 0
    outstanding = 0
    for item in db.execute("SELECT * FROM batches WHERE collected=0").fetchall():
        batch = api.batch(item["id"])
        with db:
            db.execute("UPDATE batches SET status=? WHERE id=?", (batch["status"], item["id"]))
        if batch["status"] not in FINAL_BATCH_STATES:
            outstanding += 1
            continue
        rows = {r["id"]: r for r in db.execute("SELECT * FROM images WHERE batch_id=?", (item["id"],))}
        seen = set()
        if batch.get("output_file_id"):
            for line in api.batch_content(batch["output_file_id"]).splitlines():
                result = json.loads(line)
                identity = result.get("custom_id")
                if identity not in rows:
                    continue
                seen.add(identity)
                with db:
                    body = (result.get("response") or {}).get("body")
                    if body:
                        mark_result(db, rows[identity], body, rows[identity]["model"], True)
                    else:
                        mark_error(db, rows[identity], str(result.get("error") or "No response body"))
        if batch.get("error_file_id"):
            for line in api.batch_content(batch["error_file_id"]).splitlines():
                result = json.loads(line)
                identity = result.get("custom_id")
                if identity in rows and identity not in seen:
                    seen.add(identity)
                    with db:
                        mark_error(db, rows[identity], "Batch error: " + str(result.get("error")))
        with db:
            for identity, row in rows.items():
                if identity not in seen:
                    mark_error(db, row, f"No output in terminal batch ({batch['status']})")
            db.execute("UPDATE batches SET collected=1 WHERE id=?", (item["id"],))
        finished += 1
        print(f"Collected {item['id']} ({batch['status']}).")
    db.close()
    export_files(export)
    return finished, outstanding


def md_detailed(rel: str, data: dict) -> str:
    lines = [f"## {rel}", "", data.get("overview", ""), "", "**Page style:** " + data.get("whole_page_style", ""),
             "", "**Layout:** " + data.get("layout_and_gutters", ""), ""]
    for c in data.get("cast", []):
        lines += [f"**CAST {c['id']}:** {c['design']} {c['continuity_notes']}", ""]
    for p in data["panels"]:
        lines += [f"### Panel {p['number']} | box {p['bounds_0_to_1000_xywh']}", "",
                  f"*Camera:* {p['shot_and_viewpoint']}",
                  f"*Setting:* {p['setting_and_background']}",
                  f"*Composition:* {p['composition_and_depth']}", ""]
        for a in p.get("actors", []):
            lines += [f"- **{a['id']}** — {a['position_and_scale']}; torso {a['torso_facing']}; "
                      f"head and gaze {a['head_facing_and_gaze']}; {a['pose_and_hands']}; "
                      f"expression {a['expression']}; design {a['visible_design']}; {a['interaction']}"]
        for prop in p.get("props", []):
            lines += [f"- **PROP {prop['name']}:** {prop['position']}; {prop['appearance_and_contact']}"]
        for t in p.get("text", []):
            lines += [f"- **{t['kind'].upper()} / {t['speaker_id_or_unknown']}:** "
                      f"{json.dumps(t['verbatim'], ensure_ascii=False)} — {t['placement_and_balloon_tail']}; "
                      f"{t['lettering_appearance']} [legibility: {t['legibility']}]"]
        lines += ["", f"*Beat:* {p['visible_action_and_transition']}",
                  f"*Redraw:* {p['exact_redraw_directions']}", ""]
        if p.get("uncertainties"):
            lines += ["*Uncertain:* " + "; ".join(p["uncertainties"]), ""]
    lines += ["**Final beat:** " + data.get("ending_or_final_beat", ""), "",
              "**Redraw checklist:** " + "; ".join(data.get("redraw_checklist", [])), ""]
    if data.get("global_uncertainties"):
        lines += ["**Uncertainties:** " + "; ".join(data["global_uncertainties"]), ""]
    return "\n".join(lines)


def md_abridged(rel: str, brief: dict) -> str:
    lines = [f"## {rel}", "", brief["overview"], "", f"*Style:* {brief['style']}", ""]
    for p in brief["panels"]:
        lines += [f"**Panel {p['panel']}** — {p['shot']}. {p['action']}",
                  p["blocking"], ""]
        for t in p["dialogue"]:
            lines += [f"- {t['speaker']} ({t['kind']}): {json.dumps(t['text'], ensure_ascii=False)}"]
        lines.append("")
    lines += ["**Final beat:** " + brief["ending"], ""]
    return "\n".join(lines)


def export_files(export: Path) -> None:
    db = connect(export)
    rows = db.execute("SELECT * FROM images ORDER BY relative_path").fetchall()
    # Atomic replacements for interruption-safe repeated exports.
    for field, name in (("detailed", "comprehensive.jsonl.gz"), ("abridged", "abridged.jsonl.gz")):
        tmp = export / (name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=9) as f:
            for r in rows:
                if r[field]:
                    entry = {"image_id": r["id"], "source_path": r["relative_path"],
                             "sha256": r["sha256"], "data": json.loads(r[field])}
                    f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        tmp.replace(export / name)
    for field, name, formatter in (("detailed", "comprehensive_scripts.md", md_detailed),
                                   ("abridged", "abridged_scripts.md", md_abridged)):
        tmp = export / (name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write("# " + ("Comprehensive" if field == "detailed" else "Abridged") + " comic scripts\n\n")
            for r in rows:
                if r[field]:
                    f.write(formatter(r["relative_path"], json.loads(r[field])) + "\n\n---\n\n")
        tmp.replace(export / name)
    for name, fields, selected in (
        ("status.csv", ["relative_path", "sha256", "status", "phase", "model", "attempts", "cost_usd", "input_tokens", "output_tokens", "error"], rows),
        ("review_queue.csv", ["relative_path", "reason"], [r for r in rows if r["detailed"] and review_reasons(json.loads(r["detailed"]))])):
        tmp = export / (name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            if name == "review_queue.csv":
                for r in selected:
                    w.writerow({"relative_path": r["relative_path"],
                                "reason": " | ".join(review_reasons(json.loads(r["detailed"])))})
            else:
                for r in selected:
                    w.writerow({k: r[k] for k in fields})
        tmp.replace(export / name)
    stats = {r["status"]: sum(x["status"] == r["status"] for x in rows) for r in rows}
    spent, held = spend(db)
    manifest = load_manifest(export)
    manifest["updated_at"] = now()
    manifest["stats"] = {"total": len(rows), "with_scripts": sum(bool(r["detailed"]) for r in rows),
                         "states": stats, "spent_usd": round(spent, 4), "reserved_usd": round(held, 4),
                         "tracked_max_usd": round(spent + held, 4),
                         "batches_pending": db.execute("SELECT COUNT(*) FROM batches WHERE collected=0").fetchone()[0]}
    manifest["outputs"] = ["dataset.sqlite3", "comprehensive.jsonl.gz", "abridged.jsonl.gz",
                           "comprehensive_scripts.md", "abridged_scripts.md", "status.csv", "review_queue.csv"]
    db.close()
    save_manifest(export, manifest)


def show_status(export: Path) -> None:
    export_files(export)
    m = load_manifest(export)
    print(f"Export: {export}")
    print(f"Source: {m['source_folder']}")
    print(f"Scripts: {m['stats']['with_scripts']} / {m['stats']['total']}")
    print(f"States: {m['stats']['states']}")
    print(f"Cost: ${m['stats']['spent_usd']:.2f} charged/estimated + ${m['stats']['reserved_usd']:.2f} reserved")
    print(f"Uncollected batches: {m['stats']['batches_pending']}")
    print("Data: dataset.sqlite3 | comprehensive.jsonl.gz | abridged.jsonl.gz")


def queue_refinement(export: Path, target_model: str, limit: int) -> int:
    db = connect(export)
    unfinished = db.execute("SELECT COUNT(*) FROM images WHERE status IN ('pending','submitted','sending')").fetchone()[0]
    if unfinished:
        db.close()
        raise ValueError("Finish or reconcile all pending/batch items before refinement")
    scored = []
    for r in db.execute("SELECT * FROM images WHERE status='done' AND detailed IS NOT NULL"):
        reasons = review_reasons(json.loads(r["detailed"]))
        # Prioritize uncertain text first; then dense page layouts and dialogue.
        data = json.loads(r["detailed"])
        score = len(reasons) * 100 + len(data.get("panels", [])) * 5 + sum(len(p.get("text", [])) for p in data["panels"])
        scored.append((score, r["id"]))
    scored.sort(reverse=True)
    chosen = [identity for _, identity in scored[:limit]]
    with db:
        for identity in chosen:
            db.execute("UPDATE images SET status='pending', phase='refine', model=?, error=NULL WHERE id=?",
                       (target_model, identity))
    db.close()
    # Model belongs to the run-level options so request builders use the upgrade model.
    m = load_manifest(export)
    m["options"]["model"] = target_model
    save_manifest(export, m)
    return len(chosen)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="image-dataset-generator",
        description="Turn comics/images into redraw-ready play scripts with OpenAI vision.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("configure", help="Securely prompt for OpenAI API key")
    plan = sub.add_parser("plan", help="Scan a folder and calculate conservative price reservations")
    plan.add_argument("folder", type=Path)
    plan.add_argument("-r", "--recursive", action="store_true")
    plan.add_argument("--model", choices=PRICES, default=DEFAULT_MODEL)
    plan.add_argument("--max-side", type=int, default=2048)
    plan.add_argument("--max-output-tokens", type=int, default=3600)
    plan.add_argument("--transport", choices=("batch", "live"), default="batch")
    run = sub.add_parser("run", help="Create timestamped export in current working directory and process folder")
    run.add_argument("folder", type=Path)
    run.add_argument("-r", "--recursive", action="store_true")
    run.add_argument("--model", choices=PRICES, default=DEFAULT_MODEL)
    run.add_argument("--transport", choices=("batch", "live"), default="batch")
    run.add_argument("--max-side", type=int, default=2048)
    run.add_argument("--max-output-tokens", type=int, default=3600)
    run.add_argument("--reasoning", choices=("none", "low", "medium", "high"), default="low")
    run.add_argument("--target-usd", type=float, default=49.0)
    run.add_argument("--hard-cap-usd", type=float, default=54.0)
    run.add_argument("--output-base", type=Path, default=Path.cwd())
    run.add_argument("--limit", type=int, default=0, help="Only process N images (pilot)")
    run.add_argument("--detach", action="store_true", help="Submit a batch and exit without polling")
    run.add_argument("--poll-seconds", type=int, default=60)
    resume = sub.add_parser("resume", help="Collect batches, then submit remaining images (or run live)")
    resume.add_argument("export", type=Path)
    resume.add_argument("--detach", action="store_true")
    resume.add_argument("--retry-failed", action="store_true")
    resume.add_argument("--poll-seconds", type=int, default=60)
    attach = sub.add_parser("attach-batch", help="Recover an uncertain submission using its OpenAI batch ID")
    attach.add_argument("export", type=Path)
    attach.add_argument("batch_id")
    release = sub.add_parser("release-unconfirmed", help="Reset uncertain requests ONLY after checking no batch exists")
    release.add_argument("export", type=Path)
    release.add_argument("--i-confirm-no-batch", action="store_true", required=True)
    status = sub.add_parser("status", help="Report progress and tracked spend")
    status.add_argument("export", type=Path)
    refresh = sub.add_parser("export", help="Regenerate JSONL/Markdown/CSV from SQLite without API calls")
    refresh.add_argument("export", type=Path)
    refine = sub.add_parser("refine", help="Re-analyze selected weak pages using a stronger model")
    refine.add_argument("export", type=Path)
    refine.add_argument("--model", choices=PRICES, default="gpt-6.2-astra")
    refine.add_argument("--limit", type=int, default=100)
    refine.add_argument("--detach", action="store_true")
    refine.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        print("\nQuick start: image-dataset-generator configure; image-dataset-generator run ./comics -r")
        return 0
    if args.command == "configure":
        set_key()
        return 0
    if args.command == "plan":
        if args.max_side < 256:
            raise ValueError("--max-side must be at least 256 pixels")
        files = scan_images(args.folder, args.recursive)
        count = 0
        reserve = 0.0
        for p in files:
            try:
                with Image.open(p) as im:
                    width, height = ImageOps.exif_transpose(im).size
                    ratio = min(1.0, args.max_side / max(width, height))
                    w, h = max(1, int(width * ratio)), max(1, int(height * ratio))
                    reserve += reserve_cost(args.model, args.transport == "batch", w, h,
                                            args.max_output_tokens, p.name)
                    count += 1
            except OSError as exc:
                print(f"Skipping unreadable: {p}: {exc}")
        print(f"Images: {count} | model: {args.model} | {args.transport} | original-detail, max side {args.max_side}")
        print(f"Conservative full-token reservation: ${reserve:.2f}; actual cost likely lower.")
        print("Default cost target $49; hard ceiling $54 (per-run estimates, not account billing).")
        return 0
    export = getattr(args, "export", None)
    if args.command == "run":
        if args.max_side < 256 or args.max_output_tokens < 800:
            raise ValueError("--max-side >=256 and --max-output-tokens >=800 required")
        if args.model.startswith("gpt-6.2-") and args.reasoning == "none":
            raise ValueError("GPT-6.2 models require reasoning effort low or higher")
        if not (0 < args.target_usd <= args.hard_cap_usd <= 54.0):
            raise ValueError("Require 0 < target <= hard-cap <= $54")
        options = {"recursive": args.recursive, "model": args.model,
                   "transport": args.transport, "max_side": args.max_side,
                   "max_output_tokens": args.max_output_tokens, "reasoning": args.reasoning,
                   "target_usd": args.target_usd, "hard_cap_usd": args.hard_cap_usd}
        export = create_export(args.folder, options, args.output_base)
        images = add_sources(export, args.folder.resolve(), args.recursive)
        print(f"Export: {export} | Images found: {images}")
        if not images:
            show_status(export)
            return 0
    if export is None:
        raise ValueError("Missing export directory")
    export = export.resolve()
    if args.command in ("status", "export", "release-unconfirmed"):
        if args.command == "status":
            show_status(export)
        elif args.command == "export":
            export_files(export)
            print(f"Rebuilt files in {export}")
        else:
            db = connect(export)
            with db:
                db.execute("""UPDATE images SET status='pending', batch_id=NULL,
                    reserved_usd=0, error=NULL WHERE status='sending'""")
            db.close()
            print("Reset unconfirmed submissions after explicit user confirmation.")
            show_status(export)
        return 0
    manifest = load_manifest(export)
    opts = manifest["options"]
    client = API(api_key())
    if args.command == "attach-batch":
        remote = client.batch(args.batch_id)
        file_id = remote["input_file_id"]
        db = connect(export)
        pending = db.execute("SELECT * FROM images WHERE status='sending' AND batch_id=?",
                             ("unconfirmed:" + file_id,)).fetchall()
        if not pending:
            db.close()
            raise ValueError("No unconfirmed requests match that remote batch input file")
        with db:
            db.execute("INSERT OR IGNORE INTO batches(id,input_file_id,status,submitted_at,request_count) VALUES(?,?,?,?,?)",
                       (args.batch_id, file_id, remote["status"], now(), len(pending)))
            db.execute("UPDATE images SET status='submitted', batch_id=? WHERE status='sending' AND batch_id=?",
                       (args.batch_id, "unconfirmed:" + file_id))
        db.close()
        print(f"Attached remote batch {args.batch_id} to {len(pending)} images.")
        collect_batches(export, client)
        show_status(export)
        return 0
    if args.command == "resume" and args.retry_failed:
        db = connect(export)
        with db:
            db.execute("UPDATE images SET status='pending', error=NULL WHERE status='error'")
        db.close()
    if args.command == "refine":
        if args.model.startswith("gpt-6.2-") and opts["reasoning"] == "none":
            raise ValueError("GPT-6.2 refinement requires reasoning effort low or higher")
        pending = queue_refinement(export, args.model, args.limit)
        print(f"Queued {pending} images for refined analysis with {args.model}")
        manifest = load_manifest(export)
        opts = manifest["options"]
    if opts["transport"] == "live":
        run_live(export, client, getattr(args, "limit", 0) if args.command == "run" else 0)
    else:
        # Always collect before submitting more, to release unused reservations.
        collect_batches(export, client)
        submit_batches(export, client, limit=getattr(args, "limit", 0) if args.command == "run" else 0)
        if not args.detach:
            while True:
                done, outstanding = collect_batches(export, client)
                db = connect(export)
                pending = db.execute("SELECT COUNT(*) FROM images WHERE status='pending'").fetchone()[0]
                db.close()
                if outstanding == 0 and pending:
                    if args.command == "run" and args.limit:
                        break  # pilot runs never silently expand to the entire folder
                    newly_submitted = submit_batches(export, client)
                    if not newly_submitted:
                        break  # budget exhausted or blocked image
                elif outstanding == 0:
                    break
                else:
                    print(f"Waiting on {outstanding} batch(es). Ctrl+C to stop safely; resume later.")
                    time.sleep(max(5, args.poll_seconds))
    show_status(export)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped. Existing batches are still running; use 'resume' to collect them.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError, APIError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
