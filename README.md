# 다썰어

[디스코드 봇 링크](hyperlink)

다썰어는 [Monologg님의 koElectra 모델](https://github.com/monologg/KoELECTRA)을 활용한 혐오 표현 탐지 및 분류 모델입니다. 이 깃허브에는 데이터 전처리 코드, 전이학습 코드, 모델 비교, pre-trained model, discord bot 구현을 위한 코드가 들어있습니다. transformers ver와 python ver에서 작업하였고 테스트 되었습니다.

학습할 때 사용한 디바이스는 google colab과 aws 어쩌구 저쩌구 입니다.

# architecture

<프로젝트 구조도>

# 데이터

학습할 때 사용한 데이터는 [Korean UnSmile Dataset](https://github.com/smilegate-ai/korean_unsmile_dataset?fbclid=IwAR0xTlHYCWK0LtrghSL1bPm2su69-LbjisutmcvLlERlHzroMlVpHq3h71g)과 [APEACH - Korean Hate Speech Evaluation Datasets](https://github.com/jason9693/APEACH?fbclid=IwAR2ZBPFnv8qSy1RRqISoGkTfqmitoSLz0Fma3iPv4PZJvkZo5lAm9kForo8)을 사용했습니다.

# 시연

# reference

electra
koelectra
koeran unsmile datsaet
apeach
earlystopping
