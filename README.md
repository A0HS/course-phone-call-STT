# course-phone-call-STT

PC에서 통화 MP3를 받아쓰고, OpenAI API로 미팅 기록과 To-Do를 만든 뒤 Neon PostgreSQL에 저장하는 로컬 웹앱입니다.

## 기능

- `faster-whisper large-v3-turbo` 기반 한국어 받아쓰기
- `용어집.md`를 받아쓰기 힌트로 사용
- `[분:초]` 단위 받아쓰기 결과 표시
- 거래처·요약·결정 사항 미팅 기록 생성
- 할 일·담당·기한 To-Do 생성
- 미팅 목록과 To-Do 목록 표시
- To-Do 완료 체크 및 Neon 저장
- MP3 파일은 PC에서만 STT 처리

## 요구사항

- Windows 10/11
- Python 3.10 이상
- NVIDIA CUDA GPU 권장
- FFmpeg
- OpenAI API 키
- Neon PostgreSQL 연결 문자열

## 설치

PowerShell에서 프로젝트 폴더로 이동한 뒤 실행합니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

FFmpeg가 PATH에 없으면 FFmpeg 실행 파일의 `bin` 폴더를 PATH에 추가합니다.

## 환경변수

`.env.example`을 참고해 `.env`를 만들고 값을 입력합니다.

```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=postgresql://user:password@host/database?sslmode=require
```

`OPENAI_API_KEY`는 Windows 환경변수에 설정해도 됩니다. `.env`와 API 키는 공개 저장소에 올리지 마세요.

## 실행

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

또는 다음 스크립트를 실행합니다.

```powershell
.\실행하기.ps1
```

브라우저에서 [http://127.0.0.1:8000](http://127.0.0.1:8000)을 엽니다.

첫 실행 시 `large-v3-turbo` 모델을 다운로드할 수 있습니다. Neon 연결이 되면 앱 시작 시 `schema.sql`의 `meetings`, `todos` 테이블을 자동으로 생성합니다.

## 폴더 구조

```text
app.py          # FastAPI 서버, STT, OpenAI 분석, Neon 저장
web/index.html  # 웹 화면
schema.sql      # Neon 테이블 정의
용어집.md        # STT 용어 힌트
```

녹음 파일과 받아쓰기 결과는 개인정보 보호를 위해 Git에 포함하지 않습니다.
