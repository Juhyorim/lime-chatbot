"""
corpus.jsonl -> pgvector 적재 (LlamaIndex + PGVectorStore).

*** 가장 중요한 조건 ***
lime-rag의 /chat 이 검색하는 것과 '같은 테이블 / 같은 임베딩 / 같은 인덱스'에 넣어야 한다.
아래 값들은 이미 네 main.py 의 build_vector_store() / Settings 와 동일하게 맞춰두었다:
  - table_name = "profile_chunks"  (실제 테이블은 data_profile_chunks)
  - embed model = text-embedding-3-small, embed_dim = 1536
  - chunk_size = 500, chunk_overlap = 50
  - hnsw_kwargs = main.py와 동일 (코사인 + HNSW m=16, ef_construction=64, ef_search=40)

각 문단의 title을 노드 메타데이터로 저장한다.
-> 나중에 검색 결과에서 title을 꺼내 eval_set의 gold_titles와 비교 = Hit Rate 계산.
"""
import argparse
import json
import os

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore


def load_corpus(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            doc = Document(
                text=r["text"],
                id_=r["doc_id"],
                metadata={"doc_id": r["doc_id"], "title": r["title"]},
            )
            # title이 검색용 임베딩에는 섞이지 않게 하되(순수 본문으로 검색),
            # 검색 결과 노드에서는 title을 읽을 수 있게 남겨둔다.
            doc.excluded_embed_metadata_keys = ["doc_id", "title"]
            doc.excluded_llm_metadata_keys = ["doc_id", "title"]
            docs.append(doc)
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--table", default="profile_chunks", help="main.py의 table_name과 동일")
    ap.add_argument("--embed-dim", type=int, default=1536)
    ap.add_argument("--chunk-size", type=int, default=500)   # main.py Settings와 동일
    ap.add_argument("--chunk-overlap", type=int, default=50)
    args = ap.parse_args()

    # lime-rag와 동일한 임베딩 모델이어야 검색이 일관된다.
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.node_parser = SentenceSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )

    vector_store = PGVectorStore.from_params(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        database=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        table_name=args.table,
        embed_dim=args.embed_dim,
        # main.py의 build_vector_store()와 동일하게 지정.
        # 같은 테이블에 대해 인덱스 설정이 어긋나지 않도록 반드시 맞춘다.
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    docs = load_corpus(args.corpus)
    storage = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(docs, storage_context=storage, show_progress=True)

    print(f"문단 {len(docs)}개 적재 완료 -> 테이블 data_{args.table}")


if __name__ == "__main__":
    main()