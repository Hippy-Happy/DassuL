# 다썰어

[디스코드 봇 링크](hyperlink)

다썰어는 [kiyoungkim1님의 electra-kor-base](https://github.com/kiyoungkim1/LMkor)를 활용한 혐오 표현 탐지 및 분류 모델입니다. 이 깃허브에는 데이터 전처리 코드, 전이학습 코드, 모델 비교, pre-trained model, discord bot 구현을 위한 코드가 들어있습니다. transformers ver와 python ver에서 작업하였고 테스트 되었습니다.

학습할 때 사용한 디바이스는 google colab과 aws 어쩌구 저쩌구 입니다.

# architecture

<프로젝트 구조도>

# 데이터

학습할 때 사용한 데이터는 [Korean UnSmile Dataset](https://github.com/smilegate-ai/korean_unsmile_dataset?fbclid=IwAR0xTlHYCWK0LtrghSL1bPm2su69-LbjisutmcvLlERlHzroMlVpHq3h71g)과 [APEACH - Korean Hate Speech Evaluation Datasets](https://github.com/jason9693/APEACH?fbclid=IwAR2ZBPFnv8qSy1RRqISoGkTfqmitoSLz0Fma3iPv4PZJvkZo5lAm9kForo8)을 사용했습니다. Korean UnSmile Dataset의 11가지 혐오 분류 기준에서, "여성/가족"과 "남성"을 "성별"이라는 하나의 분류로 정리하고 모델 학습에 사용했습니다. 

# 모델링 및 비교
## 모델링



## 비교

Text Classification에서 좋은 성능을 보여주고 있는 다양한 NLP 모델을 활용해서 모델링했고, 이를 `unsmile` 테스트 데이터셋을 활용해 성능을 측정하였습니다. 성능 지표는 `mean_roc_auc_score`이며, `sklearn`의 `roc_auc_score`를 활용하였습니다.

먼저, `unsmile` 데이터셋을 활용한 여러 모델들의 결과입니다.


여기서 성능이 좋았던 모델들

# 시연

# reference

electra
koelectra
roberta
koeran unsmile datsaet
apeach
earlystopping
