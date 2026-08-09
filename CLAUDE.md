\# 프로젝트: 라임 소개 RAG 챗봇 (인수인계 문서)



이 문서는 프로젝트의 현재 상태를 정리한 것이다. 이어서 작업할 때 이 맥락을 기준으로 삼는다.



\## 한 줄 요약



"라임(사용자)을 소개하는" RAG 챗봇. 방문자가 웹 화면에서 라임에 대해 질문하면,

저장된 프로필 정보를 근거로 답한다. AWS EC2 한 대 위에서 Docker Compose 로 운영한다.



\## 기술 스택



\- 백엔드: FastAPI (Python 3.11)

\- RAG 프레임워크: LlamaIndex

\- 벡터 DB: PostgreSQL + pgvector (EC2 안에서 컨테이너로 직접 운영, RDS 아님)

\- 임베딩: OpenAI `text-embedding-3-small` (1536차원) — API 호출

\- LLM: OpenAI `gpt-4o-mini` — API 호출

\- 프론트: 단일 HTML (`app/static/index.html`), 바닐라 JS

\- 리버스 프록시 / HTTPS: Caddy (Let's Encrypt 인증서 자동 발급)

\- 배포: Docker Compose (컨테이너 3개: db, app, caddy)



임베딩과 LLM 만 외부(OpenAI)로 나가고, 나머지는 전부 EC2 안에서 돈다.

그래서 GPU 없는 작은 EC2 로 충분하다.



\## 파일 구조



```

lime-rag/

├── CLAUDE.md            # 이 문서

├── docker-compose.yml   # 컨테이너 3개 정의 (db, app, caddy)

├── Dockerfile           # 앱 이미지 (python:3.11-slim)

├── Caddyfile            # 도메인 → 앱 프록시 + 자동 HTTPS

├── requirements.txt     # Python 의존성 (버전 고정됨)

├── .env.example         # 환경변수 템플릿 (.env 는 git 제외)

├── .gitignore

├── README.md            # 사용자용 실행 안내

└── app/

&#x20;   ├── main.py          # FastAPI + LlamaIndex 전체 로직

&#x20;   └── static/

&#x20;       └── index.html   # 채팅 화면

```



\## 아키텍처 / 요청 흐름



```

사용자 브라우저 ──https──> Caddy(80/443) ──> FastAPI 앱(8000)

&#x20;                                                │

&#x20;                                   ┌────────────┼────────────┐

&#x20;                                   ▼                         ▼

&#x20;                         OpenAI API(임베딩·LLM)      Postgres+pgvector(내부)

```



\- Caddy 가 유일한 외부 진입점. 80/443 만 외부 노출.

\- 앱은 `expose: 8000` 으로 컨테이너 네트워크 안에서만 열림 (외부 직접 접근 불가).

\- DB(5432)는 외부에 절대 노출하지 않음. 앱만 서비스명 `db` 로 접속.



\## 두 가지 동작 흐름



1\) 저장/업데이트 (`POST /ingest`, 토큰 보호)

&#x20;  텍스트 → 청킹(SentenceSplitter, chunk\_size=500/overlap=50) → OpenAI 임베딩 → pgvector 저장.

&#x20;  `doc\_id` 단위로 관리. 같은 doc\_id 로 다시 넣으면 기존 청크를 지우고 새로 넣음 = 갱신.



2\) 질의 (`POST /chat`)

&#x20;  질문 → OpenAI 임베딩 → pgvector 유사도 검색(top\_k=4) → 프롬프트 조립 → OpenAI LLM 응답.

&#x20;  커스텀 QA 프롬프트로 "참고 정보 안에서만, 한국어로, 없으면 지어내지 말 것" 을 강제.



\## API 엔드포인트



\- `GET  /`               채팅 화면(index.html)

\- `POST /chat`           `{"message": "..."}` → `{"answer": "..."}`

\- `POST /ingest`         헤더 `x-ingest-token` 필요. `{"doc\_id","text"}` 로 정보 저장/갱신

\- `DELETE /ingest/{doc\_id}`  헤더 `x-ingest-token` 필요. 해당 doc\_id 삭제

\- `GET  /health`         상태 확인

\- `GET  /docs`           FastAPI 자동 문서 (브라우저에서 정보 넣을 때 사용)



\## 환경변수 (.env)



```

OPENAI\_API\_KEY=      # OpenAI 키

POSTGRES\_USER=       # DB 사용자

POSTGRES\_PASSWORD=   # DB 비밀번호

POSTGRES\_DB=         # DB 이름

INGEST\_TOKEN=        # /ingest 보호용 토큰

```



\### ⚠️ .env 위치 (중요)



`.env` 는 이 프로젝트 폴더 \*\*밖(상위 폴더)\*\* 에 있다. 폴더 구조:



```

\~/lime/

├── .env               ← 비밀값. 프로젝트 폴더 밖 (이 폴더는 접근 권한 밖)

└── lime-chatbot/      ← 현재 프로젝트 루트 (작업 범위)

&#x20;   ├── CLAUDE.md

&#x20;   └── ...

```



이렇게 한 이유: 이 프로젝트 폴더에만 접근 권한을 주고, 비밀값(.env)은

권한 밖 상위 폴더에 두어 노출을 막기 위함이다.



따라서:

\- 이 폴더 안에는 `.env` 가 없다. 없다고 새로 만들지 말 것. 상위 폴더에 이미 있다.

\- docker compose 는 반드시 `--env-file ../.env` 로 상위 폴더의 .env 를 읽어야 한다.

\- `.env` 는 git 에 올리지 않는다. `.env.example` 이 템플릿(값은 비어 있음).



\## 배포 / 실행



`.env` 가 상위 폴더에 있으므로 `--env-file` 로 경로를 지정한다:



```bash

docker compose --env-file ../.env up -d --build

```



편의 스크립트 `run.sh`(프로젝트 폴더 안)를 쓰면 `--env-file` 을 자동 처리한다:



```bash

./run.sh up -d --build     # 띄우기

./run.sh logs -f caddy     # 로그

./run.sh ps                # 상태

./run.sh down              # 내리기

```



\- Caddy 가 뜨면서 Let's Encrypt 인증서를 자동 발급한다.

\- 로그 확인: `docker compose logs -f caddy` / `docker compose logs -f app`

\- 상태 확인: `docker compose ps`



\## 현재 배포 상태 (2026년 기준, 실제 값은 확인 필요)



\- 도메인: `mukbbo.mylimeorange.site` (A 레코드 → EC2 퍼블릭 IP `3.35.167.238`)

\- 인증서 발급 관련: 발급이 되려면 EC2 보안 그룹에서 80/443 이 `0.0.0.0/0` 으로 열려 있어야 한다.

&#x20; (초기에 80 이 막혀 ACME 챌린지 timeout 이 발생했었음. 보안 그룹 열어 해결하는 단계였다.)

\- 이어받을 때 먼저 `docker compose logs caddy` 로 인증서 발급 성공 여부를 확인할 것.



\## 알려진 주의점 / 히스토리



\- requirements 버전 충돌 이력: `llama-index-core` 를 낮게 고정했다가

&#x20; `llama-index-vector-stores-postgres` 가 요구하는 core 버전과 충돌해 pip 빌드가 실패했었다.

&#x20; 현재 requirements.txt 는 호환 조합으로 고정되어 있다. 업그레이드 시 조합 호환성 재확인 필요.

\- 작은 EC2(RAM 1GB) 는 빌드 중 메모리 부족으로 죽을 수 있음 → swap 2G 추가로 해결 가능.

\- HTTPS 인증서는 IP 가 아니라 도메인에만 발급된다. 80 포트가 외부에 열려 있어야 ACME 검증이 된다.

\- 현재는 질문마다 독립 응답(멀티턴 대화 기억 없음). 맥락 기억이 필요하면

&#x20; `as\_query\_engine` 대신 `as\_chat\_engine` 으로 전환해야 한다.

\- LlamaIndex 는 버전에 따라 API 가 바뀌므로, 코드 수정 전 설치된 버전 기준으로 확인할 것.



\## 앞으로 할 만한 일 (백로그)



\- \[ ] 인증서 발급 최종 확인 및 https 접속 검증

\- \[ ] `/docs` 로 라임 프로필 정보(about, career, hobby 등) 실제 적재

\- \[ ] 백업 자동화: `pg\_dump` → S3 업로드를 cron 에 등록

\- \[ ] (선택) 멀티턴 대화(chat\_engine) 전환

\- \[ ] (선택) `/ingest` 관리를 위한 간단한 관리자 화면

\- \[ ] (선택) 대화 로그 저장 테이블 추가



\## 코드 이어받을 때 원칙



\- DB(5432)를 외부에 노출하는 변경은 하지 말 것.

\- 비밀값(.env, API 키)을 코드나 git 에 하드코딩하지 말 것.

\- 임베딩 모델을 바꾸면 차원(EMBED\_DIM)과 테이블을 함께 맞춰야 한다(기존 벡터와 차원 불일치 주의).

