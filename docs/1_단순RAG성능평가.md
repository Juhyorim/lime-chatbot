### 첫 번째 실험

```javascript
{
  "tag": "v0_baseline",
  "top_k": 4,
  "n_questions": 500,
  "hit_rate": 0.96,
  "recall": 0.766,
  "mrr": 0.9015
}
```

- 테이블에 데이터를 중복저장(2번씩)함

### 두 번째 실험

{
"tag": "v0_diag",
"top_k": 4,
"n_questions": 500,
"hit_rate": 0.97,
"recall": 0.791,
"mrr": 0.9062
}

[1] top-4 슬롯 안의 서로 다른 문서 수
평균 4.00개 (top_k=4 대비)
→ 이 값이 top_k보다 크게 작으면, 한 문서가 여러 청크로
슬롯을 채워 Hit/MRR을 올리고 Recall을 누르는 상태.
4개 문서: 500

[2] level 별 지표 (누수라면 hard까지 균일하게 높음 → 의심 신호)

```
    level         n      hit     recall    mrr
    hard          500    0.970   0.791    0.906

    type 별 (bridge=멀티홉, comparison=비교)
    type            n      hit   recall      mrr
    bridge        388    0.964    0.745    0.895
    comparison    112    0.991    0.951    0.946
```

### 세 번째 실험 - hit@1

{
"tag": "v0_hit1",
"retrieve_k": 4,
"eval_k": 4,
"top_k": 4,
"n_questions": 500,
"hit@1": 0.86,
"hit_rate": 0.97,
"recall": 0.791,
"mrr": 0.9062,
"ndcg": 0.7849
}

[1] eval-4 슬롯 안의 서로 다른 문서 수
평균 4.00개 (eval_k=4 대비)
→ 이 값이 eval_k보다 크게 작으면, 한 문서가 여러 청크로
슬롯을 채워 Hit/MRR을 올리고 Recall을 누르는 상태.
4개 문서: 500

[2] level 별 지표 (누수라면 hard까지 균일하게 높음 → 의심 신호)

```
                    n    hit@1      hit   recall      mrr     ndcg
    hard          500    0.860    0.970    0.791    0.906    0.785

    type 별 (bridge=멀티홉, comparison=비교)
                    n    hit@1      hit   recall      mrr     ndcg
    bridge        388    0.845    0.964    0.745    0.895    0.745
    comparison    112    0.911    0.991    0.951    0.946    0.924
```

### 네 번째 실험 k=20

{
"tag": "v0_recall20",
"retrieve_k": 20,
"eval_k": 20,
"top_k": 20,
"n_questions": 500,
"hit@1": 0.86,
"hit_rate": 1.0,
"recall": 0.938,
"mrr": 0.9103,
"ndcg": 0.8424
}

[1] eval-20 슬롯 안의 서로 다른 문서 수
평균 19.97개 (eval_k=20 대비)
→ 이 값이 eval_k보다 크게 작으면, 한 문서가 여러 청크로
슬롯을 채워 Hit/MRR을 올리고 Recall을 누르는 상태.
19개 문서: 14
20개 문서: 486

[2] level 별 지표 (누수라면 hard까지 균일하게 높음 → 의심 신호)

```
                    n    hit@1      hit   recall      mrr     ndcg
    hard          500    0.860    1.000    0.938    0.910    0.842

    type 별 (bridge=멀티홉, comparison=비교)
                    n    hit@1      hit   recall      mrr     ndcg
    bridge        388    0.845    1.000    0.920    0.900    0.813
    comparison    112    0.911    1.000    1.000    0.946    0.944
```

- hit_rate 1.0...원하던 결과가 아니다..
