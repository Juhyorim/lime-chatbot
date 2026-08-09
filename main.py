"""
라임 소개 RAG 챗봇 - FastAPI + LlamaIndex + pgvector

두 가지 흐름을 담당한다.
  1) 저장/업데이트 흐름 : POST /ingest  (내 정보 텍스트를 청킹→임베딩→pgvector에 저장)
  2) 질의 흐름        : POST /chat    (질문→임베딩→유사도 검색→LLM 응답)
정보는 doc_id 단위로 관리되어, 같은 doc_id로 다시 넣으면 기존 내용을 지우고 갱신한다.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore

# --------------------------------------------------------------------------
# 모델 설정 (임베딩 + LLM 은 둘 다 OpenAI API 로 호출)
# --------------------------------------------------------------------------
EMBED_DIM = 1536  # text-embedding-3-small 의 차원

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.3)
# 문서를 자를 청킹 규칙 (약 500토큰, 50토큰 겹침)
Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)

# 응답 톤을 잡아주는 프롬프트: 참고 정보 안에서만, 한국어로, 지어내지 않기
QA_PROMPT = PromptTemplate(
    "당신은 '라임'이라는 사람을 소개하는 친근한 어시스턴트입니다.\n"
    "아래 '참고 정보'에 있는 내용만 근거로 사용자 질문에 한국어로 자연스럽게 답하세요.\n"
    "참고 정보에 없는 내용은 추측하거나 지어내지 말고, "
    "'그 부분은 제가 아직 모르는 내용이에요'라고 답하세요.\n"
    "---------------------\n"
    "참고 정보:\n{context_str}\n"
    "---------------------\n"
    "질문: {query_str}\n"
    "답변: "
)

state: dict = {}


def build_vector_store() -> PGVectorStore:
    """환경변수를 읽어 pgvector 연결을 만든다. HNSW 인덱스 + 코사인 유사도."""
    return PGVectorStore.from_params(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        table_name="profile_chunks",
        embed_dim=EMBED_DIM,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버가 뜰 때 한 번: 벡터 스토어에 연결하고 기존 인덱스를 로드한다."""
    vector_store = build_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    state["vector_store"] = vector_store
    state["index"] = index
    yield
    # 종료 시 정리
    vector_store.close()


app = FastAPI(title="라임 소개 챗봇", lifespan=lifespan)


# --------------------------------------------------------------------------
# 요청 스키마
# --------------------------------------------------------------------------
class ChatIn(BaseModel):
    message: str


class IngestIn(BaseModel):
    doc_id: str  # 정보 조각의 이름 (예: "about", "career", "hobby")
    text: str    # 그 조각의 원문


# --------------------------------------------------------------------------
# 질의 흐름 : 사용자가 질문할 때마다
# --------------------------------------------------------------------------
@app.post("/chat")
async def chat(body: ChatIn):
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="빈 메시지입니다.")

    query_engine = state["index"].as_query_engine(
        similarity_top_k=4,          # 가장 비슷한 청크 4개를 근거로
        text_qa_template=QA_PROMPT,
    )
    response = await query_engine.aquery(message)
    return {"answer": str(response)}


# --------------------------------------------------------------------------
# 저장/업데이트 흐름 : 내 정보를 넣거나 고칠 때 (토큰으로 보호)
# def(동기)로 두어 FastAPI 가 스레드풀에서 처리 → 이벤트 루프를 막지 않음
# --------------------------------------------------------------------------
@app.post("/ingest")
def ingest(body: IngestIn, x_ingest_token: str = Header(default="")):
    if x_ingest_token != os.environ.get("INGEST_TOKEN", ""):
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

    index: VectorStoreIndex = state["index"]
    vector_store: PGVectorStore = state["vector_store"]

    # 같은 doc_id 로 들어온 기존 청크를 먼저 지우고 (없으면 무시) 새로 넣는다 → 갱신
    try:
        vector_store.delete(ref_doc_id=body.doc_id)
    except Exception:
        pass

    document = Document(text=body.text, doc_id=body.doc_id)
    index.insert(document)
    return {"status": "ok", "doc_id": body.doc_id}


@app.delete("/ingest/{doc_id}")
def delete_doc(doc_id: str, x_ingest_token: str = Header(default="")):
    if x_ingest_token != os.environ.get("INGEST_TOKEN", ""):
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
    state["vector_store"].delete(ref_doc_id=doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.get("/health")
def health():
    return {"status": "healthy"}


# --------------------------------------------------------------------------
# 정적 화면 (채팅 HTML)
# --------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def root():
    return FileResponse("app/static/index.html")
