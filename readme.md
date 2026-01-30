- 기존에는 데이터가 라벨링이 되어있지 않은 점을 고려하여 Decoder의 Prior(시계열에서 정상일 때 기대되는 연결 구조)가 Encoder에서 학습한 (series association:실제 데이터로부터 학습된 연결 구조) 을 기준으로 삼도록 설계했다.

- 하지만 이러한 경우 Encoder가 잘못 학습되었을 때 Decoder 또한 함께 잘못 학습되며, AD의 역할을 충분히 활용하지 못하게 된다는 부작용이 있다.

- 따라서 Decoder의 Prior를 기존에 이용하던 series-based prior와 명시적인 prior를 hybrid로 사용하도록 수정해본다.
