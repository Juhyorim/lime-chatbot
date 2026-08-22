# HotpotQA -> lime-rag 수집 파이프라인

HotpotQA 서브셋을 lime-rag의 pgvector에 넣는 2단계 파이프라인.
값들은 이미 네 main.py(profile_chunks / 1536 / chunk 500·50 / HNSW cosine)에 맞춰져 있다.

## 역할 분담 (중요)

- prepare_hotpotqa.py : DB 불필요. huggingface 접근만 필요 -> 호스트에서 실행.
- ingest_to_pgvector.py : DB 필요. 5432가 외부 비노출이므로 -> 앱 컨테이너 안에서 실행.
  (ingest는 datasets 불필요. 앱에 이미 있는 llama-index만 쓴다.)

## 1) 전처리 (호스트에서)

```bash
pip install datasets
python prepare_hotpotqa.py --n 500
# 결과: corpus.jsonl, eval_set.jsonl
```

## 2) 적재 (앱 컨테이너 안에서)

```bash
# corpus.jsonl 을 컨테이너로 복사 (서비스명이 app 이라고 가정)
docker compose cp corpus.jsonl app:/app/corpus.jsonl
docker compose cp ingest_to_pgvector.py app:/app/ingest_to_pgvector.py

# 컨테이너 안에서 실행 (POSTGRES_*, OPENAI_API_KEY 는 컨테이너 env에서 상속됨)
docker compose exec app python /app/ingest_to_pgvector.py --corpus /app/corpus.jsonl
```

## 참고

- 적재 후 앱 재시작 불필요: /chat 의 as_query_engine 은 매 요청마다 벡터스토어를
  직접 조회하므로 새 데이터가 즉시 반영된다.
- 재적재(다시 돌리기) 시에는 중복이 쌓인다. 먼저 테이블을 비울 것:
  docker compose exec db psql -U $POSTGRES_USER -d $POSTGRES_DB \
   -c 'TRUNCATE data_profile_chunks;'
- eval_set.jsonl 은 다음 단계(평가)에서 사용. 검색 결과 노드의 metadata["title"] 을
  각 질문의 gold_titles 와 대조하면 Hit Rate 가 나온다.
