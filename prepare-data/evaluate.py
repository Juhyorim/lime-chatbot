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
    # --- 진단 옵션 (기존 지표 계산에는 영향 없음) ---
    ap.add_argument("--diagnose", action="store_true",
                    help="진단 모드: top-k 내 distinct 문서 수 / level·type별 분해 출력")
    ap.add_argument("--dump", type=int, default=0,
                    help="진단 시 눈으로 확인할 샘플 질문 개수 (0=안 함). 검색된 청크 본문·gold 여부 표시")
    ap.add_argument("--no-record", action="store_true",
                    help="results.jsonl에 기록하지 않음 (진단만 돌릴 때 결과 오염 방지)")
    args = ap.parse_args()

    index = build_index()
    retriever = index.as_retriever(similarity_top_k=args.top_k)
    rows = load_eval(args.eval)
    n = len(rows)

    hit_sum = recall_sum = mrr_sum = 0.0
    em_sum = f1_sum = 0.0
    gen_n = 0
    per_rows = []  # 진단 모드에서만 채워짐 (질문별 세부 기록)

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

        # --- 검색 지표 (계산값은 원래와 동일, 변수로만 분리) ---
        retrieved_set = set(retrieved_ids)
        hit = 1.0 if (gold_ids & retrieved_set) else 0.0
        rec = len(gold_ids & retrieved_set) / max(1, len(gold_ids))
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in gold_ids:
                rr = 1.0 / rank
                break
        hit_sum += hit
        recall_sum += rec
        mrr_sum += rr

        # --- 진단용 세부 기록 (--diagnose 일 때만) ---
        if args.diagnose:
            # top-k 슬롯이 실제로 서로 다른 문서 몇 개를 보고 있는지.
            # 청킹 때문에 한 문서가 여러 청크로 top-k를 채우면 이 값이 top_k보다 작아진다.
            distinct_ids = [rid for rid in retrieved_set if rid is not None]
            per_rows.append({
                "level": row.get("level", "unknown"),
                "type": row.get("type", "unknown"),
                "hit": hit,
                "recall": rec,
                "rr": rr,
                "n_distinct": len(distinct_ids),
                "n_gold": len(gold_ids),
                # 샘플 덤프용 (필요할 때만 사용)
                "question": q,
                "gold_ids": sorted(gold_ids),
                "retrieved_ids": retrieved_ids,
                "snippets": [nd.node.get_content()[:120].replace("\n", " ") for nd in nodes],
            })

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

    if not args.no_record:
        with open(args.results, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.diagnose:
        print_diagnosis(per_rows, args.top_k, args.dump)


def _agg(rows):
    """행 리스트 -> (n, hit, recall, mrr) 평균."""
    m = len(rows)
    if m == 0:
        return 0, 0.0, 0.0, 0.0
    return (
        m,
        sum(r["hit"] for r in rows) / m,
        sum(r["recall"] for r in rows) / m,
        sum(r["rr"] for r in rows) / m,
    )


def print_diagnosis(per_rows, top_k, dump):
    """baseline이 '진짜'인지 눈으로 확인하기 위한 진단 출력.
    핵심 질문 3개에 답한다:
      (1) top-k가 실제로 서로 다른 문서 몇 개를 보고 있나? (청킹 캡 확인)
      (2) 쉬운 층위/어려운 층위가 제대로 구분되나? (누수면 전부 균일하게 높음)
      (3) 검색된 청크에 답이 실제로 들어있나? (--dump 로 눈 확인)
    """
    print("\n" + "=" * 60)
    print("진단 (DIAGNOSIS)")
    print("=" * 60)

    # (1) top-k 내 distinct 문서 수 분포
    from collections import Counter
    dist = Counter(r["n_distinct"] for r in per_rows)
    avg_distinct = sum(r["n_distinct"] for r in per_rows) / max(1, len(per_rows))
    print(f"\n[1] top-{top_k} 슬롯 안의 서로 다른 문서 수")
    print(f"    평균 {avg_distinct:.2f}개  (top_k={top_k} 대비)")
    print(f"    → 이 값이 top_k보다 크게 작으면, 한 문서가 여러 청크로")
    print(f"      슬롯을 채워 Hit/MRR을 올리고 Recall을 누르는 상태.")
    for k in sorted(dist):
        bar = "█" * dist[k]
        print(f"    {k}개 문서: {dist[k]:4d}  {bar}")

    # (2) level / type 별 분해
    def by_key(key):
        buckets = {}
        for r in per_rows:
            buckets.setdefault(r[key], []).append(r)
        return buckets

    print(f"\n[2] level 별 지표  (누수라면 hard까지 균일하게 높음 → 의심 신호)")
    print(f"    {'level':<10}{'n':>5}{'hit':>9}{'recall':>9}{'mrr':>9}")
    for lv, rs in sorted(by_key("level").items()):
        m, h, rc, mr = _agg(rs)
        print(f"    {lv:<10}{m:>5}{h:>9.3f}{rc:>9.3f}{mr:>9.3f}")

    print(f"\n    type 별 (bridge=멀티홉, comparison=비교)")
    print(f"    {'type':<12}{'n':>5}{'hit':>9}{'recall':>9}{'mrr':>9}")
    for tp, rs in sorted(by_key("type").items()):
        m, h, rc, mr = _agg(rs)
        print(f"    {tp:<12}{m:>5}{h:>9.3f}{rc:>9.3f}{mr:>9.3f}")

    # (3) 샘플 덤프
    if dump > 0:
        print(f"\n[3] 샘플 {dump}개 (질문 / 검색 문서 / gold / 청크 앞부분)")
        for r in per_rows[:dump]:
            mark = "HIT " if r["hit"] else "MISS"
            print(f"\n  [{mark}] recall={r['recall']:.2f}  {r['question']}")
            print(f"        gold:      {r['gold_ids']}")
            for rid, snip in zip(r["retrieved_ids"], r["snippets"]):
                star = " *" if rid in set(r["gold_ids"]) else "  "
                print(f"      {star} {str(rid):<40} | {snip}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()