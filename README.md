# RS HW2

2026713004 / Minseok Jeong

[SKKU ESW4024 추천시스템 과제](https://www.kaggle.com/competitions/skku-esw-4024-recommender-system/overview): Q1은 EASE, Q2는 NeuMF를 구현했다.

## 실행

대회에서 제공한 `train.csv`, `test.csv`, `sample_submission.csv`를 `data/`에 넣는다.
실험 설정은 [configs/experiment.json](configs/experiment.json), 패키지 목록은 [requirements.txt](requirements.txt)에 있다.
현재 서버의 Slurm 환경에서 코드와 설정을 커밋한 뒤 실행한다.

```bash
bash scripts/submit.sh --smoke  # 시험 실행
bash scripts/submit.sh          # 전체 실험
```

## 결과

사용자별 마지막 20% 상호작용을 시간순으로 분리해 검증하고, 선택한 설정으로 전체 학습 데이터를 다시 학습했다.

| 모델 | Validation NDCG@20 | Kaggle Public NDCG@20 |
|---|---:|---:|
| EASE | 0.155008 | 0.11419 |
| NeuMF | 0.131471 | 0.09812 |

## 파일

- [rs_hw2/](rs_hw2/): 데이터 처리, 모델, 평가 코드
- [results/](results/), [logs/](logs/): 실험 결과와 실행 기록
- [submissions/](submissions/): 제출 CSV와 Kaggle 점수 기록

참고: [EASE 논문](https://arxiv.org/abs/1905.03375), [NeuMF 논문](https://arxiv.org/abs/1708.05031), [NeuMF 원저자 구현](https://github.com/hexiangnan/neural_collaborative_filtering).
