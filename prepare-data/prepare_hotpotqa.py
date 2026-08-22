"""
HotpotQA -> lime-rag 파이프라인용 코퍼스 + 평가셋 변환기.

출력 파일 2개:
  corpus.jsonl    : 벡터DB에 넣을 위키 문단들   {doc_id, title, text}
  eval_set.jsonl  : 평가용 질문-정답-근거문서   {question, answer, gold_titles, gold_doc_ids, ...}

HotpotQA(distractor) 예제 하나의 구조:
  question, answer, type("bridge"/"comparison"), level("easy"/"medium"/"hard")
  supporting_facts = {"title": [...], "sent_id": [...]}          # 정답 근거 문장이 든 문단
  context          = {"title": [...], "sentences": [[...], ...]} # gold 2개 + distractor 8개

핵심:
  - context 안의 모든 문단(gold + distractor)을 코퍼스에 넣는다.
    distractor를 포함해야 "비슷하지만 틀린 문서" 사이에서 정답을 고르는 실전 난이도가 생긴다.
  - supporting_facts의 title이 그 질문의 정답 문서다. 이게 Hit Rate 계산의 gold 라벨이 된다.
"""
import argparse
import json
import re
from collections import defaultdict

from datasets import load_dataset


def normalize_doc_id(title: str) -> str:
    """위키 title을 안정적인 doc_id로 변환 (공백->_, 특수문자 제거)."""
    slug = re.sub(r"\s+", "_", title.strip())
    slug = re.sub(r"[^0-9A-Za-z_\-가-힣]", "", slug)
    return slug or "untitled"


def build(rows):
    corpus = {}       # doc_id -> {doc_id, title, text}  (title 기준 자동 dedup)
    eval_rows = []

    for row in rows:
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]

        # context의 모든 문단을 코퍼스에 추가
        for title, sents in zip(titles, sentences):
            doc_id = normalize_doc_id(title)
            if doc_id not in corpus:
                corpus[doc_id] = {
                    "doc_id": doc_id,
                    "title": title,
                    "text": " ".join(s.strip() for s in sents).strip(),
                }

        # supporting_facts.title = 이 질문의 정답(gold) 문서
        gold_titles = sorted(set(row["supporting_facts"]["title"]))
        gold_doc_ids = sorted({normalize_doc_id(t) for t in gold_titles})

        eval_rows.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "gold_titles": gold_titles,
            "gold_doc_ids": gold_doc_ids,
            "type": row.get("type"),
            "level": row.get("level"),
        })

    return list(corpus.values()), eval_rows


def sample_balanced(ds, n, seed):
    """level(easy/medium/hard)별로 고르게 n개 샘플링.
    난이도가 섞여 있어야 개선 기법별 효과가 층위별로 잘 구분된다."""
    ds = ds.shuffle(seed=seed)
    by_level = defaultdict(list)
    for row in ds:
        by_level[row.get("level", "unknown")].append(row)

    levels = list(by_level.keys())
    per = max(1, n // len(levels))
    picked = []
    for lv in levels:
        picked.extend(by_level[lv][:per])
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="샘플링할 질문 수 (시작은 500 권장)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus-out", default="corpus.jsonl")
    ap.add_argument("--eval-out", default="eval_set.jsonl")
    args = ap.parse_args()

    # distractor 설정: 질문마다 gold 2 + distractor 8 문단.
    # validation split은 라벨(supporting_facts)이 공개돼 있어 평가에 바로 쓸 수 있다.
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    rows = sample_balanced(ds, args.n, args.seed)

    corpus, eval_rows = build(rows)

    with open(args.corpus_out, "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(args.eval_out, "w", encoding="utf-8") as f:
        for e in eval_rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"질문 {len(eval_rows)}개 -> 코퍼스 문단 {len(corpus)}개 (dedup 후)")
    print(f"저장: {args.corpus_out}, {args.eval_out}")


if __name__ == "__main__":
    main()