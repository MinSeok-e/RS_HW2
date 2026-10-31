# RS HW2

2026713004 / Minseok Jeong

Kaggle: [SKKU ESW4024 Recommender System](https://www.kaggle.com/competitions/skku-esw-4024-recommender-system/overview)

## 과제와 실험 상태

Q1은 EASE, Q2는 NeuMF로 수행한다. 실행 결과와 Kaggle 점수는 실제 실행 후 기록한다.
공통 데이터 처리, 평가, 제출 생성과 Slurm 실행 코드를 포함한다.

## 실행 환경

Python 3.10, NumPy 1.26.4, SciPy 1.13.1, PyTorch 2.3.1 (CUDA 12.1), Matplotlib 3.8.4.
이 서버에서는 `/home/minseok/.venvs/rs-hw2`를 사용한다. 이 venv는 기존 사용자 환경
`/home/minseok/.conda/envs/bot`의 패키지를 읽기 전용으로 참조한다. 기존 환경은 수정하지 않았다.
정확한 실행 버전과 경로는 실행별 `manifest.json`, 전체 패키지는 `logs/environment-freeze.txt`에 남긴다.
독립 환경을 재현할 때는 계산 노드에서 `environment.yml` 또는 `requirements.txt`를 사용한다.

**로그인 노드에서는 Python, 설치, 학습, 테스트를 실행하지 않는다. 모든 실행은 Slurm 계산 노드에서 한다.**

## 데이터와 검증 설계

- 대회가 제공한 `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`만 사용한다.
- 원본 MovieLens를 추가 다운로드하거나 사용자/영화 ID를 외부 데이터와 연결하지 않는다.
- 제공된 train.csv의 ID를 정렬하여 내부 인덱스로 매핑한다. 이 단계에서는 ID의 존재만 사용한다.
- 모든 평점을 관측 상호작용 1로 변환한다. 미관측을 확정적인 비선호라고 가정하지 않는다.
- 사용자별 `time_idx` 오름차순으로 마지막 `ceil(0.2*n)`개를 검증에 두고 나머지를 학습한다.
- validation은 사용자별 binary NDCG@20의 평균이다. 보조 지표로 Recall@20과 Hit@20을 기록한다.
- 학습에서 한 번 이상 나타난 모든 영화를 후보로 평가하며, 해당 사용자의 학습 영화를 제외한다.
- 검증에서만 등장한 영화는 모델이 학습할 수 없으므로 추천에서 제외하되, 정답 수/IDCG에는 포함한다.
  따라서 cold item 정답을 삭제하여 점수를 높이지 않는다. 관련 수치는 `data_summary.json`에 기록한다.
- 점수가 같으면 원래 movieId가 작은 순서로 추천한다. 테스트 id와 userId는 별도로 매핑한다.
- Kaggle 숨김 정답의 분할/관련성 세부 규칙은 공개 안내에 없으므로 위 기준은 명시적인 로컬 검증 가정이다.
- `time_idx`는 분할에 사용한다. 이 HW2의 두 모델은 순차 모델이 아니다.

## 모델

### EASE

`min ||X-XB||_F^2 + lambda ||B||_F^2`, `diag(B)=0`을 푼다.
정확한 해를 Woodbury 항등식으로 사용자 공간에서 계산하여 큰 아이템 정방행렬을 만들지 않는다.
`A=(XX^T+lambda I)^-1`, `Pdiag=(1-diag(X^TAX))/lambda`일 때
학습 사용자에 대한 점수는 `X - AX/Pdiag`이다. 작은 행렬에서 원래 아이템 공간 해와 일치하는지 테스트한다.
정규화 계수 `[1, 10, 30, 100, 300, 1000]` 중 검증 NDCG@20으로 선택한다.

### NeuMF

별도의 사용자/아이템 임베딩을 사용하는 GMF(element-wise product)와 MLP(ReLU)를 결합하고
선형 출력의 logit에 `BCEWithLogitsLoss`를 적용한다. 사전학습 없이 처음부터 학습한다.
학습 positive마다 fit-active 영화 중 그 사용자의 학습 positive가 아닌 영화를 균일 복원추출한다.
Negative sampling에서 validation 정답은 조회하지 않는다. 미관측 중 일부가 실제 선호일 가능성은 남는다.
Adam을 사용하며 weight_decay 설정은 전체 파라미터에 적용되는 L2 페널티다.
최대 100 epoch, 검증 NDCG@20이 10 epoch 연속 개선되지 않으면 중단한다.
같은 시드 2020으로 `configs/experiment.json`의 6개 설정을 비교한다.
최적 설정을 선택한 뒤 제공된 train.csv 전체로 다시 학습하며 학습 epoch 수는 검증 최적 epoch로 고정한다.

## 실행

저장소의 코드/설정을 먼저 커밋한다. 각 Slurm 작업은 지정한 커밋을 `snapshots/`에 추출해 실행하므로
대기 중 또는 실행 중 작업 디렉터리를 편집해도 실제 실행 소스가 바뀌지 않는다.

```bash
# 로그인 노드: 커밋된 코드 제출
bash scripts/submit.sh --smoke
bash scripts/submit.sh
```

`slurm/experiment.sbatch`는 GPU 1개, CPU 4개, RAM 6GB, 최대 2시간을 요청한다.
시험 실행은 `sbatch --time=00:10:00 ... --smoke`로 짧게 요청할 수도 있다.
실행 전 9개 핵심 테스트를 수행한다. 모든 모델과 설정은 같은 split SHA256을 공유한다.

## 산출물

- `results/<run_id>/`: 설정, 데이터 해시, 코드 커밋, 패키지 버전, 전체 로그, 실험 결과와 그래프.
- `checkpoints/<run_id>/`: 검증/최종 모델. Git 추적 제외.
- `submissions/<run_id>_{ease,neumf}.csv`: 모델별 최종 제출 파일. smoke 파일은 대회에 제출하지 않는다.
- `logs/`: Slurm 표준 출력/오류와 환경 기록.
- `report/`: 보고서 원본 및 PDF. Kaggle 점수·스크린샷 확인 후 최종화한다.

제출 파일은 588개 테스트 행의 id 순서, 중복 없는 영화 20개, 원래 movieId 복원,
학습에서 이미 본 영화의 제외를 자동 검사한다. 데이터가 달라지면 행 수는 실제 test.csv에 맞춘다.

## 출처

- Harald Steck. *Embarrassingly Shallow Autoencoders for Sparse Data*, WWW 2019. https://arxiv.org/abs/1905.03375
- Xiangnan He et al. *Neural Collaborative Filtering*, WWW 2017. https://arxiv.org/abs/1708.05031
- NeuMF 원저자 구현(모델 구성 참고): https://github.com/hexiangnan/neural_collaborative_filtering

본 저장소의 모델은 위 수식/구조를 바탕으로 새로 작성한 NumPy/PyTorch 구현이다.
원저자 구현의 sampled-negative HR/NDCG@10 평가 대신 이 과제의 공통 full-catalog NDCG@20 평가를 사용한다.
