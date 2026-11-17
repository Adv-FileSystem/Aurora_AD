# Aurora_AD
AuroraFS에서 시스템 이상 탐지(Anomaly Detection)를 위한 알고리즘 개발 프로젝트

- 기존에는 데이터가 라벨링이 되어있지 않은 점을 고려하여 Decoder의 Prior(시계열에서 정상일 때 기대되는 연결 구조)가 Encoder에서 학습한 (series association:실제 데이터로부터 학습된 연결 구조) 을 기준으로 삼도록 설계했다.

- 하지만 이러한 경우 Encoder가 잘못 학습되었을 때 Decoder 또한 함께 잘못 학습되며, AD의 역할을 충분히 활용하지 못하게 된다는 부작용이 있다.

- 따라서 Decoder의 Prior를 기존에 이용하던 series-based prior와 명시적인 prior를 hybrid로 사용하도록 수정해본다.



- 기존에는 residual boost가 `y_reg_q[q0.5]`에서 Δ를 뽑아 별도의 cumsum 재구성 경로로 `yhat` 만들었음

- 그런데 cls 타깃/평가에 쓰는 건 decode_autoregressive로 만든 `y_pred`라서 경로가 다름

- 이때, residual boost는 그냥 `y_pred`(실제 autoreg 예측 결과)로 residual score 계산

- 따라서 Δ를 절대값처럼 쓰는 구간 제거가 필요함



- consistency trend break 감지 : 시계열의 장기적 패턴(추세·주기)이 유지되던 일관성을 벗어나는 시점을 이상으로 감지하는 보지 지표

- variance shift score 도입 : 신호의 평균은 유지되지만 분산(변동성)이 급격히 변화하는 구간을 수치화해 이상 징후로 포착하는 지표

- MLP 기반 anomaly score fusion 도입 :
→ 다양한 이상 단서(어텐션 기반 AD score, 예측 오차 기반 residual feature 등)를 비선형적으로 결합하여 최종 이상 점수를 산출하는 MLP 모듈을 도입.


### TEST RESULTS (2026.02.09)

| Horizon (k) | PAD_AUPRC | F1_adj | Threshold |
|------------|---------|-----------|-----------|
| 1          | 0.990478  | 0.7353 | 0.535199  |
| 60         | 0.987602  | 0.7324 | 0.596985  |

