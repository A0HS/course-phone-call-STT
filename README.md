# 통화 STT · 미팅 기록

FastAPI 기반 통화 받아쓰기 앱입니다. 내 PC에서는 `faster-whisper large-v3-turbo`를 사용하고, Vercel 또는 `STT_ENGINE=openai` 모드에서는 OpenAI `whisper-1` 받아쓰기 API를 사용합니다. 두 모드 모두 구간별 받아쓰기, 미팅 요약, To-Do, Neon 저장 형식이 같습니다.

## 내 PC에서 실행

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-local.txt
.\실행하기.ps1
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다. 로컬 모드는 faster-whisper 모델을 처음 실행할 때 내려받습니다. FFmpeg가 PATH에 있어야 합니다.

## 환경변수

`.env.example`을 참고해 `.env`를 만들 수 있습니다.

- `OPENAI_API_KEY`: 서버 키가 있으면 웹 화면의 키 입력칸을 숨기고 이 키를 사용합니다.
- `STT_ENGINE=openai`: 로컬에서도 웹(OpenAI) 엔진을 테스트합니다. `VERCEL`이 있으면 자동으로 OpenAI 모드입니다.
- `DATABASE_URL` 또는 `NEON_DATABASE_URL`: 설정된 경우에만 Neon에 저장합니다. 없으면 화면에 결과만 표시합니다.
- `APP_ACCESS_CODE`: 설정하면 화면에서 접근 코드를 입력해야 API가 동작합니다.
- `OPENAI_MODEL`: 미팅 요약 모델. 기본값은 `gpt-4o-mini`입니다.

웹 모드에서 서버 키가 없으면 사용자가 입력한 OpenAI 키가 브라우저 `localStorage`에만 저장되고 요청 헤더로 전달됩니다. 키는 서버 로그·응답·DB에 저장하지 않습니다. 웹 업로드는 Vercel 제한에 맞춰 4.5MB 이하만 허용합니다.

## Vercel

이 폴더를 배포하고 다음 환경변수를 Vercel 프로젝트에 설정합니다.

```text
OPENAI_API_KEY
OPENAI_MODEL              # 선택
DATABASE_URL              # 선택
APP_ACCESS_CODE           # 선택
STT_ENGINE=openai         # 선택; VERCEL이 있으면 자동으로 openai
```

`.env`, 녹음 파일, 결과 파일은 저장소에 포함하지 않습니다.
