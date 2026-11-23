from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from openai import OpenAI


load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
GLOSSARY_PATH = BASE_DIR / "용어집.md"
RECORDINGS_DIR = BASE_DIR / "녹음"
WEB_DIR = BASE_DIR / "web"
SCHEMA_PATH = BASE_DIR / "schema.sql"

_model: WhisperModel | None = None
_model_config: dict[str, str] = {}


def load_glossary() -> list[str]:
    text = GLOSSARY_PATH.read_text(encoding="utf-8") if GLOSSARY_PATH.exists() else ""
    return re.findall(r"^\s*-\s+(.+?)\s*$", text, flags=re.MULTILINE)


GLOSSARY_TERMS = load_glossary()


def get_model() -> WhisperModel:
    global _model, _model_config
    if _model is None:
        try:
            import torch
            use_cuda = bool(torch.cuda.is_available())
        except Exception:
            use_cuda = False
        device = "cuda" if use_cuda else "cpu"
        compute_type = "float16" if use_cuda else "int8"
        _model_config = {"device": device, "compute_type": compute_type}
        _model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
    return _model


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"[{minutes:02d}:{secs:02d}]"


def infer_recording_date(filename: str, fallback: date | None = None) -> date:
    # Prefer a date in the filename, e.g. *_1119.mp3 or *_20261119.mp3.
    match = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", filename)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
    match = re.search(r"(?<!\d)(\d{2})(\d{2})(?!\d)", Path(filename).stem)
    if match:
        year = (fallback or date.today()).year
        try:
            candidate = date(year, int(match.group(1)), int(match.group(2)))
            if candidate > (fallback or date.today()) + timedelta(days=31):
                candidate = date(year - 1, int(match.group(1)), int(match.group(2)))
            return candidate
        except ValueError:
            pass
    return fallback or date.today()


def transcribe_path(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    model = get_model()
    prompt = "다음 용어의 표기와 띄어쓰기를 우선 사용하세요: " + ", ".join(GLOSSARY_TERMS)
    segments, info = model.transcribe(
        str(path), language="ko", task="transcribe", beam_size=5,
        initial_prompt=prompt, vad_filter=True, condition_on_previous_text=True,
    )
    rows = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            rows.append({
                "start": segment.start, "end": segment.end,
                "timestamp": format_timestamp(segment.start), "text": text,
            })
    return {
        "filename": path.name,
        "recording_date": infer_recording_date(path.name, datetime.fromtimestamp(path.stat().st_mtime).date()).isoformat(),
        "duration": getattr(info, "duration", None),
        "language": getattr(info, "language", "ko"),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "device": _model_config.get("device", "unknown"),
        "compute_type": _model_config.get("compute_type", "unknown"),
        "segments": rows,
    }


MEETING_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "client": {"type": "string"},
        "summary": {"type": "string"},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "todos": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"task": {"type": "string"}, "assignee": {"type": "string"}, "due_date": {"type": ["string", "null"]}},
            "required": ["task", "assignee", "due_date"]}},
    },
    "required": ["client", "summary", "decisions", "todos"],
}


def analyze_transcript(result: dict[str, Any]) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    transcript = "\n".join(f"{row['timestamp']} {row['text']}" for row in result["segments"])
    meeting_date = result["recording_date"]
    client = OpenAI().responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        input=[
            {"role": "system", "content": "당신은 한국어 영업 통화 회의록 정리 담당자입니다. 녹취에 없는 내용을 추측하지 마세요. 거래처명, 요약, 결정 사항, To-Do를 정확히 추출하세요. 기한이 상대 표현이면 미팅 날짜를 기준으로 YYYY-MM-DD로 계산하고, 없으면 null로 하세요."},
            {"role": "user", "content": f"미팅 날짜: {meeting_date}\n\n녹취:\n{transcript}"},
        ],
        text={"format": {"type": "json_schema", "name": "meeting_record", "strict": True, "schema": MEETING_SCHEMA}},
    )
    data = json.loads(client.output_text)
    return {"client": data["client"], "summary": data["summary"], "decisions": data["decisions"], "todos": data["todos"]}


def database_url() -> str | None:
    return os.getenv("DATABASE_URL") or os.getenv("NEON_DATABASE_URL")


def init_db() -> None:
    url = database_url()
    if not url or not SCHEMA_PATH.exists():
        return
    with psycopg.connect(url) as conn:
        conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()


def save_to_db(result: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    url = database_url()
    if not url:
        return {"saved": False, "reason": "DATABASE_URL이 없어 로컬 화면에만 표시했습니다."}
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO meetings (recording_date, filename, client, summary, decisions, transcript, duration_seconds, processing_seconds)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (result["recording_date"], result["filename"], analysis["client"], analysis["summary"], json.dumps(analysis["decisions"], ensure_ascii=False),
                 json.dumps(result["segments"], ensure_ascii=False), result.get("duration"), result.get("elapsed_seconds")),
            )
            meeting_id = cur.fetchone()[0]
            for todo in analysis["todos"]:
                cur.execute("INSERT INTO todos (meeting_id, task, assignee, due_date) VALUES (%s,%s,%s,%s)", (meeting_id, todo["task"], todo["assignee"], todo["due_date"]))
        conn.commit()
    return {"saved": True, "meeting_id": meeting_id}


def process_path(path: Path, display_filename: str | None = None) -> dict[str, Any]:
    result = transcribe_path(path)
    if display_filename:
        result["filename"] = display_filename
        result["recording_date"] = infer_recording_date(display_filename).isoformat()
    try:
        analysis = analyze_transcript(result)
        result["analysis"] = analysis
        result["database"] = save_to_db(result, analysis)
    except Exception as exc:
        result["analysis_error"] = str(exc)
        result["database"] = {"saved": False, "reason": "미팅 기록 생성/저장 실패"}
    return result


app = FastAPI(title="로컬 받아쓰기 미팅 기록", docs_url=None, redoc_url=None)


@app.on_event("startup")
def startup() -> None:
    try:
        init_db()
    except Exception as exc:
        print(f"Database initialization skipped: {exc}")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/samples")
def samples() -> list[dict[str, Any]]:
    if not RECORDINGS_DIR.exists():
        return []
    return [{"name": p.name, "path": p.name, "size": p.stat().st_size} for p in sorted(RECORDINGS_DIR.iterdir()) if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".webm", ".mp4"}]


@app.post("/api/transcribe-upload")
async def transcribe_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "audio.mp3").suffix.lower()
    if suffix not in {".mp3", ".wav", ".m4a", ".webm", ".mp4", ".ogg", ".flac"}:
        raise HTTPException(status_code=400, detail="지원하지 않는 오디오 형식입니다.")
    temp_path = BASE_DIR / f".stt-upload{suffix}"
    try:
        temp_path.write_bytes(await file.read())
        return process_path(temp_path, file.filename or temp_path.name)
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/transcribe-sample")
def transcribe_sample(filename: str = Body(...)) -> dict[str, Any]:
    candidate = (RECORDINGS_DIR / filename).resolve()
    if RECORDINGS_DIR.resolve() not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="녹음 폴더에서 파일을 찾을 수 없습니다.")
    return process_path(candidate)


@app.get("/api/meetings")
def meetings() -> list[dict[str, Any]]:
    url = database_url()
    if not url:
        return []
    with psycopg.connect(url) as conn:
        rows = conn.execute("SELECT id, recording_date, filename, client, summary, decisions, created_at FROM meetings ORDER BY recording_date DESC, id DESC").fetchall()
    return [{"id": r[0], "recording_date": r[1].isoformat(), "filename": r[2], "client": r[3], "summary": r[4], "decisions": r[5], "created_at": r[6].isoformat()} for r in rows]


@app.get("/api/todos")
def todos() -> list[dict[str, Any]]:
    url = database_url()
    if not url:
        return []
    with psycopg.connect(url) as conn:
        rows = conn.execute("SELECT t.id, t.task, t.assignee, t.due_date, t.completed, m.client, m.filename FROM todos t JOIN meetings m ON m.id=t.meeting_id ORDER BY t.completed, t.due_date NULLS LAST, t.id DESC").fetchall()
    return [{"id": r[0], "task": r[1], "assignee": r[2], "due_date": r[3].isoformat() if r[3] else None, "completed": r[4], "client": r[5], "filename": r[6]} for r in rows]


@app.patch("/api/todos/{todo_id}")
def update_todo(todo_id: int, payload: dict[str, bool]) -> dict[str, Any]:
    url = database_url()
    if not url:
        raise HTTPException(status_code=503, detail="DATABASE_URL이 설정되어 있지 않습니다.")
    with psycopg.connect(url) as conn:
        conn.execute("UPDATE todos SET completed=%s WHERE id=%s", (payload.get("completed", False), todo_id))
        conn.commit()
    return {"ok": True}
