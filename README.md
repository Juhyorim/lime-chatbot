# 라임 소개 RAG 챗봇

FastAPI + LlamaIndex + pgvector(Postgres) 로 만든 "나를 소개하는" RAG 챗봇.
EC2 한 대 안에서 Docker Compose 로 DB와 앱을 함께 띄운다.

## 구조

```
lime-rag/
├── docker-compose.yml     # Postgres(pgvector) + 앱 컨테이너
├── Dockerfile             # 앱 이미지
├── requirements.txt       # Python 의존성
├── .env.example           # 환경변수 템플릿 (복사해서 .env 로 사용)
├── app/
│   ├── main.py            # FastAPI + LlamaIndex 로직
│   └── static/
│       └── index.html     # 채팅 화면
```

동작 흐름은 두 가지다.

- 저장/업데이트: 텍스트 → 청킹 → OpenAI 임베딩 → pgvector 저장 (`POST /ingest`)
- 질의: 질문 → 임베딩 → 유사도 검색(상위 4개) → OpenAI 응답 (`POST /chat`)

정보는 `doc_id` 단위로 관리된다. 같은 `doc_id` 로 다시 넣으면 기존 내용을 지우고 새로 넣으므로, 자주 업데이트하기 좋다.

## 1. 준비

```bash
cp .env.example .env
# .env 를 열어 OPENAI_API_KEY, 비밀번호, INGEST_TOKEN 을 채운다
```

## 2. 실행

```bash
docker compose up -d --build
```

- 앱이 80번 포트로 뜬다. 브라우저에서 `http://<EC2-공개주소>/` 로 접속.
- DB(5432)는 외부에 노출하지 않는다. 앱 컨테이너만 접근한다.

> EC2 보안 그룹에서 80(그리고 HTTPS 를 붙이면 443)만 열고, 5432 는 절대 열지 말 것.

## 3. 내 정보 넣기 (초기 적재 · 업데이트 공용)

`INGEST_TOKEN` 을 헤더에 넣어 호출한다. `doc_id` 는 정보 조각의 이름이다.

```bash
curl -X POST http://<EC2-공개주소>/ingest \
  -H "Content-Type: application/json" \
  -H "x-ingest-token: <당신의_INGEST_TOKEN>" \
  -d '{
    "doc_id": "about",
    "text": "라임은 백엔드 개발을 공부하는 사람이다. 새로운 걸 만드는 걸 좋아한다."
  }'
```

취미, 경력 등 조각을 나눠 여러 번 넣으면 된다.

```bash
curl -X POST http://<EC2-공개주소>/ingest \
  -H "Content-Type: application/json" \
  -H "x-ingest-token: <당신의_INGEST_TOKEN>" \
  -d '{"doc_id": "hobby", "text": "취미는 등산과 커피 내리기다."}'
```

정보를 고치려면 같은 `doc_id` 로 다시 넣으면 덮어써진다.
조각을 지우려면:

```bash
curl -X DELETE http://<EC2-공개주소>/ingest/hobby \
  -H "x-ingest-token: <당신의_INGEST_TOKEN>"
```

## 4. 대화

브라우저로 접속해서 질문하거나, 직접 호출해도 된다.

```bash
curl -X POST http://<EC2-공개주소>/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "라임은 어떤 사람이야?"}'
```

## 5. 백업 (꼭 챙기기)

데이터는 `pgdata` 볼륨에 남지만, EC2 자체가 사라지면 함께 사라진다.
주기적으로 덤프를 떠서 S3 같은 곳에 올려두는 것을 권장한다.

```bash
# 덤프 만들기
docker compose exec db pg_dump -U <POSTGRES_USER> <POSTGRES_DB> > backup.sql

# (선택) S3 업로드 — awscli 가 설정돼 있다면
aws s3 cp backup.sql s3://<버킷>/lime-rag/backup-$(date +%F).sql
```

cron 에 걸어 매일 자동 백업하면 안전하다.

## + 참고

- 임베딩 모델은 `text-embedding-3-small`(1536차원). 바꾸면 `main.py` 의 `EMBED_DIM` 도 함께 바꿔야 하고, 기존 테이블을 새로 만들어야 한다.
- LLM 은 `gpt-4o-mini`. 더 좋은 품질을 원하면 `main.py` 의 모델명을 바꾸면 된다.
- 멀티턴 대화(맥락 기억)가 필요해지면 `as_query_engine` 대신 `as_chat_engine` 으로 확장할 수 있다.
- LlamaIndex 는 버전에 따라 API 가 바뀌므로, 업그레이드할 땐 공식 문서를 확인할 것. `requirements.txt` 는 버전을 고정해 두었다.

## 테스트 방법

- HotpotQA의 작은 서브셋 사용 -> 파이프라인 안정화 & baseline 측정
