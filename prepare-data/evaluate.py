"""
lime-rag baseline 평가 (평가 전용 스크립트).
실행 중인 /chat API를 거치지 않고 벡터스토어를 '직접' 조회한다.
-> 검색된 문단의 doc_id 를 꺼내 정답 라벨과 대조할 수 있어 검색 지표가 정확하고, HTTP 왕복이 없어 빠르다.

측정 지표:
  [검색] Hit Rate@k : gold 문서 중 하나라도 top-k 안에 있으면 1        (빠르고 결정적)
         Recall@k   : top-k 안에 든 gold 문서 비율 (HotpotQA는 gold가 보통 2개)
         MRR        : 첫 gold 문서가 나온 순위의 역수
  [생성] Answer EM  : 생성 답과 정답이 정확히 일치 (HotpotQA 공식 정규화)
         Answer F1  : 생성 답과 정답의 단어 겹침 F1
  ※ 생성 지표는 --generate 옵션일 때만. API 호출이라 느리므로 --limit 로 서브셋만 권장.

결과는 results.jsonl 에 --tag 와 함께 append 된다. 실험 버전(v0, v1, ...)별 비교 기록이 쌓인다.
"""
import argparse
import json
import os
import re
import string
from collections import Counter

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore

EMBED_DIM = 1536

# 평가용 중립 프롬프트. main.py의 '라임 소개용' 프롬프트와 분리한다.
# HotpotQA는 짧은 정답을 원하므로 간결하게 답하도록 유도.
EVAL_QA_PROMPT = PromptTemplate(
    "Answer the question using ONLY the context below. "
    "Reply with the shortest exact answer (a name, an entity, or yes/no).\n"
    "If the answer is not in the context, reply 'unknown'.\n"
    "---------------------\n"
    "Context:\n{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)


def build_index():
    """main.py의 build_vector_store()와 동일 설정으로 같은 테이블에 연결."""
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)  # 평가는 재현성 위해 0
    vector_store = PGVectorStore.from_params(
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
    storage = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage)


# ---- HotpotQA 공식 answer 정규화 + EM / F1 ----
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1(pred, gold):
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    same = sum((Counter(p_toks) & Counter(g_toks)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p_toks), same / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def load_eval(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval_set.jsonl")
    ap.add_argument("--top-k", type=int, default=4, help="main.py의 similarity_top_k와 동일")
    ap.add_argument("--tag", default="v0_baseline", help="실험 버전 이름 (기록용)")
    ap.add_argument("--generate", action="store_true", help="생성 지표(EM/F1)도 측정 (느림·API비용)")
    ap.add_argument("--limit", type=int, default=0, help="생성 평가할 질문 수 (0=전체)")
    ap.add_argument("--results", default="results.jsonl")
    args = ap.parse_args()

    index = build_index()
    retriever = index.as_retriever(similarity_top_k=args.top_k)
    rows = load_eval(args.eval)
    n = len(rows)

    hit_sum = recall_sum = mrr_sum = 0.0
    em_sum = f1_sum = 0.0
    gen_n = 0

    query_engine = None
    if args.generate:
        query_engine = index.as_query_engine(
            similarity_top_k=args.top_k, text_qa_template=EVAL_QA_PROMPT
        )

    for row in rows:
        q = row["question"]
        gold_ids = set(row["gold_doc_ids"])

        nodes = retriever.retrieve(q)
        retrieved_ids = [nd.node.metadata.get("doc_id") for nd in nodes]

        # --- 검색 지표 ---
        hit_sum += 1.0 if (gold_ids & set(retrieved_ids)) else 0.0
        recall_sum += len(gold_ids & set(retrieved_ids)) / max(1, len(gold_ids))
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in gold_ids:
                rr = 1.0 / rank
                break
        mrr_sum += rr

        # --- 생성 지표 (옵션) ---
        if query_engine is not None and (args.limit == 0 or gen_n < args.limit):
            ans = str(query_engine.query(q))
            em_sum += exact_match(ans, row["answer"])
            f1_sum += f1(ans, row["answer"])
            gen_n += 1

    result = {
        "tag": args.tag,
        "top_k": args.top_k,
        "n_questions": n,
        "hit_rate": round(hit_sum / n, 4),
        "recall": round(recall_sum / n, 4),
        "mrr": round(mrr_sum / n, 4),
    }
    if gen_n:
        result["gen_n"] = gen_n
        result["answer_em"] = round(em_sum / gen_n, 4)
        result["answer_f1"] = round(f1_sum / gen_n, 4)

    with open(args.results, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()