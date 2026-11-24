from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openai import APIStatusError, AuthenticationError, OpenAI

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
GLOSSARY_PATH = BASE_DIR / "용어집.md"
RECORDINGS_DIR = BASE_DIR / "녹음"
WEB_DIR = BASE_DIR / "web"
SCHEMA_PATH = BASE_DIR / "schema.sql"
SUPPORTED_SUFFIXES = {".mp3", ".wav", ".m4a", ".webm", ".mp4", ".ogg", ".flac"}
VERCEL_UPLOAD_LIMIT = 4_500_000

_model: Any = None
_model_config: dict[str, str] = {}


def engine_name() -> str:
    return "openai" if os.getenv("VERCEL") or os.getenv("STT_ENGINE", "").lower() == "openai" else "faster-whisper"


def load_glossary() -> list[str]:
    text = GLOSSARY_PATH.read_text(encoding="utf-8") if GLOSSARY_PATH.exists() else ""
    return re.findall(r"^\s*-\s+(.+?)\s*$", text, flags=re.MULTILINE)


GLOSSARY_TERMS = load_glossary()


def require_access_code(x_access_code: str | None = Header(default=None)) -> None:
    expected = os.getenv("APP_ACCESS_CODE")
    if expected and x_access_code != expected:
        raise HTTPException(status_code=401, detail="접근 코드가 필요하거나 올바르지 않습니다.")


def request_api_key(x_openai_api_key: str | None = Header(default=None)) -> str | None:
    server_key = os.getenv("OPENAI_API_KEY")
    if server_key:
        return server_key
    return x_openai_api_key


def get_model() -> Any:
    global _model, _model_config
    if _model is None:
        from faster_whisper import WhisperModel
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
    fallback = fallback or date.today()
    match = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", filename)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
    match = re.search(r"(?<!\d)(\d{2})(\d{2})(?!\d)", Path(filename).stem)
    if match:
        try:
            candidate = date(fallback.year, int(match.group(1)), int(match.group(2)))
            return date(fallback.year - 1, candidate.month, candidate.day) if candidate > fallback + timedelta(days=31) else candidate
        except ValueError:
            pass
    return fallback


def infer_counterparty(filename: str) -> str | None:
    match = re.match(r"^영업통화_(.+?)_(?:20\d{6}|\d{4})$", Path(filename).stem)
    return match.group(1).strip() if match else None


def openai_transcribe(path: Path, api_key: str) -> tuple[list[dict[str, Any]], float | None, str]:
    prompt = "다음 용어의 표기와 띄어쓰기를 우선 사용하세요: " + ", ".join(GLOSSARY_TERMS)
    try:
        with path.open("rb") as audio_file:
            response = OpenAI(api_key=api_key).audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="verbose_json", language="ko", prompt=prompt
            )
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail="OpenAI 키가 올바르지 않습니다. 키를 확인해 주세요.") from exc
    except APIStatusError as exc:
        if exc.status_code == 401:
            raise HTTPException(status_code=401, detail="OpenAI 키가 올바르지 않습니다. 키를 확인해 주세요.") from exc
        raise
    rows = []
    for segment in (getattr(response, "segments", None) or []):
        text = (getattr(segment, "text", "") or "").strip()
        if text:
            start = float(getattr(segment, "start", 0) or 0)
            rows.append({"start": start, "end": float(getattr(segment, "end", start) or start), "timestamp": format_timestamp(start), "text": text})
    return rows, getattr(response, "duration", None), getattr(response, "language", "ko") or "ko"


def transcribe_path(path: Path, api_key: str | None) -> dict[str, Any]:
    started = time.perf_counter()
    if engine_name() == "openai":
        if not api_key:
            raise HTTPException(status_code=401, detail="OpenAI 키를 입력해 주세요.")
        rows, duration, language = openai_transcribe(path, api_key)
        device, compute_type = "openai", "whisper-1"
    else:
        model = get_model()
        prompt = "다음 용어의 표기와 띄어쓰기를 우선 사용하세요: " + ", ".join(GLOSSARY_TERMS)
        segments, info = model.transcribe(str(path), language="ko", task="transcribe", beam_size=5, initial_prompt=prompt, vad_filter=True, condition_on_previous_text=True)
        rows = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                rows.append({"start": segment.start, "end": segment.end, "timestamp": format_timestamp(segment.start), "text": text})
        duration, language = getattr(info, "duration", None), getattr(info, "language", "ko")
        device, compute_type = _model_config.get("device", "unknown"), _model_config.get("compute_type", "unknown")
    return {"filename": path.name, "recording_date": infer_recording_date(path.name, datetime.fromtimestamp(path.stat().st_mtime).date()).isoformat(), "duration": duration, "language": language, "elapsed_seconds": round(time.perf_counter() - started, 2), "device": device, "compute_type": compute_type, "segments": rows}


MEETING_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {"client": {"type": "string"}, "summary": {"type": "string"}, "decisions": {"type": "array", "items": {"type": "string"}}, "todos": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"task": {"type": "string"}, "assignee": {"type": "string"}, "due_date": {"type": ["string", "null"]}}, "required": ["task", "assignee", "due_date"]}}}, "required": ["client", "summary", "decisions", "todos"]}


def analyze_transcript(result: dict[str, Any], api_key: str | None) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    transcript = "\n".join(f"{row['timestamp']} {row['text']}" for row in result["segments"])
    counterparty_hint = result.get("counterparty_hint") or "파일명과 녹취에서 확인"
    try:
        response = OpenAI(api_key=api_key).responses.create(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), input=[
            {"role": "system", "content": "당신은 한국어 영업 통화 회의록 정리 담당자입니다. 우리 회사는 한결오피스입니다. 거래처(client)는 반드시 한결오피스가 아닌 통화 상대 회사명을 넣으세요. 녹취에 없는 내용을 추측하지 마세요. 거래처명, 요약, 결정 사항, To-Do를 정확히 추출하세요. 기한이 상대 표현이면 미팅 날짜를 기준으로 YYYY-MM-DD로 계산하고, 없으면 null로 하세요."},
            {"role": "user", "content": f"미팅 날짜: {result['recording_date']}\n파일명에서 추정한 통화 상대 회사: {counterparty_hint}\n\n녹취:\n{transcript}"},
        ], text={"format": {"type": "json_schema", "name": "meeting_record", "strict": True, "schema": MEETING_SCHEMA}})
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail="OpenAI 키가 올바르지 않습니다. 키를 확인해 주세요.") from exc
    data = json.loads(response.output_text)
    return {"client": counterparty_hint if result.get("counterparty_hint") else data["client"], "summary": data["summary"], "decisions": data["decisions"], "todos": data["todos"]}


def database_url() -> str | None:
    return os.getenv("DATABASE_URL") or os.getenv("NEON_DATABASE_URL")


def init_db() -> None:
    if database_url() and SCHEMA_PATH.exists():
        with psycopg.connect(database_url()) as conn:
            conn.execute(SCHEMA_PATH.read_text(encoding="utf-8")); conn.commit()


def save_to_db(result: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    if not database_url():
        return {"saved": False, "reason": "DATABASE_URL이 없어 결과만 표시합니다."}
    with psycopg.connect(database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO meetings (recording_date, filename, client, summary, decisions, transcript, duration_seconds, processing_seconds) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id", (result["recording_date"], result["filename"], analysis["client"], analysis["summary"], json.dumps(analysis["decisions"], ensure_ascii=False), json.dumps(result["segments"], ensure_ascii=False), result.get("duration"), result.get("elapsed_seconds")))
            meeting_id = cur.fetchone()[0]
            for todo in analysis["todos"]:
                cur.execute("INSERT INTO todos (meeting_id, task, assignee, due_date) VALUES (%s,%s,%s,%s)", (meeting_id, todo["task"], todo["assignee"], todo["due_date"]))
        conn.commit()
    return {"saved": True, "meeting_id": meeting_id}


def process_path(path: Path, api_key: str | None, display_filename: str | None = None) -> dict[str, Any]:
    result = transcribe_path(path, api_key)
    if display_filename:
        result["filename"], result["recording_date"] = display_filename, infer_recording_date(display_filename).isoformat()
    result["counterparty_hint"] = infer_counterparty(result["filename"])
    try:
        result["analysis"] = analyze_transcript(result, api_key)
        result["database"] = save_to_db(result, result["analysis"])
    except HTTPException:
        raise
    except Exception:
        result["analysis_error"] = "미팅 기록 생성/저장에 실패했습니다."
        result["database"] = {"saved": False, "reason": "미팅 기록 생성/저장 실패"}
    return result


app = FastAPI(title="통화 STT 미팅 기록", docs_url=None, redoc_url=None)


@app.on_event("startup")
def startup() -> None:
    try: init_db()
    except Exception as exc: print(f"Database initialization skipped: {type(exc).__name__}")


@app.get("/")
def index() -> FileResponse: return FileResponse(WEB_DIR / "index.html")


@app.get("/api/config", dependencies=[Depends(require_access_code)])
def config() -> dict[str, Any]:
    return {"engine": engine_name(), "server_key_configured": bool(os.getenv("OPENAI_API_KEY")), "database_configured": bool(database_url()), "upload_limit_bytes": VERCEL_UPLOAD_LIMIT if engine_name() == "openai" else None}


@app.get("/api/samples", dependencies=[Depends(require_access_code)])
def samples() -> list[dict[str, Any]]:
    if not RECORDINGS_DIR.exists(): return []
    return [{"name": p.name, "path": p.name, "size": p.stat().st_size} for p in sorted(RECORDINGS_DIR.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]


@app.post("/api/transcribe-upload", dependencies=[Depends(require_access_code)])
async def transcribe_upload(file: UploadFile = File(...), api_key: str | None = Depends(request_api_key)) -> dict[str, Any]:
    suffix = Path(file.filename or "audio.mp3").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES: raise HTTPException(status_code=400, detail="지원하지 않는 오디오 형식입니다.")
    data = await file.read()
    if engine_name() == "openai" and len(data) > VERCEL_UPLOAD_LIMIT: raise HTTPException(status_code=413, detail="웹 모드에서는 4.5MB 이하 파일만 올릴 수 있습니다.")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="stt-upload-", suffix=suffix, dir="/tmp" if os.name != "nt" else None, delete=False) as temp:
            temp.write(data); temp_path = Path(temp.name)
        return process_path(temp_path, api_key, file.filename or temp_path.name)
    finally:
        if temp_path: temp_path.unlink(missing_ok=True)


@app.post("/api/transcribe-sample", dependencies=[Depends(require_access_code)])
def transcribe_sample(filename: str = Body(...), api_key: str | None = Depends(request_api_key)) -> dict[str, Any]:
    candidate = (RECORDINGS_DIR / filename).resolve()
    if RECORDINGS_DIR.resolve() not in candidate.parents or not candidate.is_file(): raise HTTPException(status_code=404, detail="녹음 폴더에서 파일을 찾을 수 없습니다.")
    return process_path(candidate, api_key)


@app.get("/api/meetings", dependencies=[Depends(require_access_code)])
def meetings() -> list[dict[str, Any]]:
    if not database_url(): return []
    with psycopg.connect(database_url()) as conn:
        rows = conn.execute("SELECT id, recording_date, filename, client, summary, decisions, created_at FROM meetings ORDER BY recording_date DESC, id DESC").fetchall()
    return [{"id": r[0], "recording_date": r[1].isoformat(), "filename": r[2], "client": r[3], "summary": r[4], "decisions": r[5], "created_at": r[6].isoformat()} for r in rows]


@app.get("/api/todos", dependencies=[Depends(require_access_code)])
def todos() -> list[dict[str, Any]]:
    if not database_url(): return []
    with psycopg.connect(database_url()) as conn:
        rows = conn.execute("SELECT t.id, t.task, t.assignee, t.due_date, t.completed, m.client, m.filename FROM todos t JOIN meetings m ON m.id=t.meeting_id ORDER BY t.completed, t.due_date NULLS LAST, t.id DESC").fetchall()
    return [{"id": r[0], "task": r[1], "assignee": r[2], "due_date": r[3].isoformat() if r[3] else None, "completed": r[4], "client": r[5], "filename": r[6]} for r in rows]


@app.patch("/api/todos/{todo_id}", dependencies=[Depends(require_access_code)])
def update_todo(todo_id: int, payload: dict[str, bool]) -> dict[str, Any]:
    if not database_url(): raise HTTPException(status_code=503, detail="DATABASE_URL이 설정되어 있지 않습니다.")
    with psycopg.connect(database_url()) as conn:
        conn.execute("UPDATE todos SET completed=%s WHERE id=%s", (payload.get("completed", False), todo_id)); conn.commit()
    return {"ok": True}
