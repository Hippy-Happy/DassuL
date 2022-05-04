# 다썰어

[디스코드 봇 초대 링크](https://discord.com/oauth2/authorize?client_id=964031115612536902&permissions=8&scope=bot)

다썰어는 [kiyoungkim1님의 electra-kor-base](https://github.com/kiyoungkim1/LMkor)를 활용한 혐오 표현 탐지 및 분류 모델입니다. 이 깃허브에는 데이터 전처리 코드, 전이학습 코드, 모델 비교, pre-trained model, discord bot 구현을 위한 코드가 들어있습니다. 필요한 라이브러리는 requirements.txt에 정리했습니다.

학습할 때 사용한 디바이스는 `google colab - Tesla T4`와 `aws - Tesla K80`입니다.

# 1. Architecture

![image](https://user-images.githubusercontent.com/33687740/166632121-b0b59517-a0f1-4605-8e81-72aa046c19db.png)


# 2. 데이터

- 학습할 때 사용한 데이터는 [Korean UnSmile Dataset](https://github.com/smilegate-ai/korean_unsmile_dataset?fbclid=IwAR0xTlHYCWK0LtrghSL1bPm2su69-LbjisutmcvLlERlHzroMlVpHq3h71g)과 [APEACH - Korean Hate Speech Evaluation Datasets](https://github.com/jason9693/APEACH?fbclid=IwAR2ZBPFnv8qSy1RRqISoGkTfqmitoSLz0Fma3iPv4PZJvkZo5lAm9kForo8)을 사용했습니다. 
- Korean UnSmile Dataset의 11가지 혐오 분류 기준에서, "여성/가족"과 "남성"을 "성별"이라는 하나의 분류로 정리하고 모델 학습에 사용했습니다. 상세 데이터 내용은 링크를 통해 확인하시기 바랍니다.
- `unsmile` 데이터를 카테고리별로 나누고, label의 비율을 50:50으로 구성하여 Dataset을 만들었습니다. 이 Dataset을 8:2로 나누어 trian, test로 나누었고, 다시 train을 train, valid로 나누었습니다.
- `apeach` 데이터의 카테고리는 `unsmile`과 같지 않습니다. [Meta pseudo Labels](https://arxiv.org/pdf/2003.10580v4.pdf)에서 아이디어를 얻어, 따라서 `unsmile` 데이터만을 활용한 모델을 통해 `apeach` 데이터를 predicted하고, 이를 검수하는 과정을 통해 두 데이터의 category를 똑같이 만들어서 학습에 활용하였습니다. 자세한 내용은 3.1. 모델링 부분에서 설명드리겠습니다.


# 3. 모델링 및 비교
## 3.1. 모델링
- 모든 사람이 공통적으로 인정하는 혐오 표현도 있는 반면, 혐오의 정도가 약할 경우, 또는 분류자의 가치관에 따라 분류하는 기준이 다릅니다.
- 특히, 혐오 데이터들은 혐오 분류자가 전부 다르기 때문에, 같은 혐오 문장이라도 어느 데이터셋에서는 혐오라고 분류할 수 있고, 아닐 수도 있습니다.
- 따라서, 일관적인 혐오의 분류를 위해 하나의 데이터셋을 기준으로 모델링하고, 추가적인 혐오 도메인을 모델에 전이학습 하는 형태로 구성하였습니다.
- 또한, 데이터의 카테고리 역시 데이터셋마다 다르기 때문에, [Meta pseudo Labels](https://arxiv.org/pdf/2003.10580v4.pdf)의 아이디어를 활용하였습니다.
- 하나의 데이터셋을 기준으로 모델링한 후, 다른 데이터셋에는 pseudo-label을 적용해 다시 모델에 학습시켰습니다.
- 또한, 사용한 모델들의 토크나이저가 꽤 오래되어서, 새로 생긴 혐오단어나 의미가 변형된 혐오단어는 잘 토크나이징하지 못하는 것을 확인하였습니다.
- 그래서 세번째 모델에는 사전을 추가한 토크나이저를 활용해 성능을 높였습니다.
### 3.1.1. First model
- 각 카테고리 별로 binary-classification model을 생성했습니다. 이 모델들은 각 카테고리의 혐오표현이 있으면 1, 없으면 0을 반환합니다.
- 모델들은 전부 [huggingface의 bert](https://huggingface.co/docs/transformers/main/en/model_doc/bert#bert)를 보고, 마지막 layer에 classification layer를 추가했습니다. 이 layer는 각 모델의 마지막 output만큼을 입력으로 받고, 출력으로 1개의 값을 반환합니다.
- 모델의 성능을 최대한 높이기 위해 [조기종료 기법](https://github.com/Bjarten/early-stopping-pytorch)을 사용했습니다. 
- 모델이 너무 빠르게 수렴했기 때문에, `lr`을 1/10으로 줄였습니다.
- `loss_fn`을 [BCEWithLogitLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)를 사용했습니다. 이 함수는 출력으로 `sigmoid`를 하지 않아도 함수 내에서 취하기 때문에, 좀 더 편리하게 사용할 수 있습니다. 
- 나머지 파라미터는 default값과 동일합니다.
### 3.1.2. Second model
- [Meta pseudo Labels](https://arxiv.org/pdf/2003.10580v4.pdf)의 아이디어를 활용해, `kakao` 데이터를 pseudo-labeling을 해주었습니다.
- 원 논문에서는 Teacher model보다 훨씬 작은 parameter를 갖는 student model을 활용해 pseudo-labeled data를 학습했지만, DassuL에서는 Teacher model을 전이학습하는 형태로 모델링했습니다.
  - 혐오 표현의 특징 벡터는 데이터셋마다 매우 다릅니다. 이는 데이터셋을 설계한 혐오 분류자의 domain이 각각 다르기 때문입니다.
  - 따라서, 일반적인 혐오 표현의 특징 벡터를 위해 student model을 만들지 않고, Teacher model에 전이학습을 해주었습니다.
- pseudo-labeled된 데이터에는 `unsmiled` 데이터의 편향으로 인해 잘못 분류된 데이터가 존재할 확률이 있습니다.
- 따라서, 수작업으로 이를 정제하여 데이터셋을 만들고, 이를 `First model`에 다시 재학습 시켰습니다.
### 3.1.3. Third model
- huggingface에 올라온 토크나이저는 대부분 2~3년전의 데이터를 바탕으로 만들었습니다.
- 따라서 새로 생긴 혐오단어, 욕설 등이나, 의미가 변형된 단어는 제대로 토크나이징하지 못하는 현상을 확인할 수 있었습니다.
- 한국어 토크나이저의 대부분은 지도학습을 활용한 토크나이저여서 같은 문제점을 공유하고 있습니다.
- 비지도학습을 활용한 한국어 토크나이저인 [soynlp](https://github.com/lovit/soynlp)가 있습니다.
  - 다만, 비지도학습의 특성상 데이터의 수가 많아야 좋은 성능을 기대할 수 있습니다.
  - 또한, 혐오표현의 특성상 그 토큰이 희소하고 매우 많이 변화해 제대로 토큰화하지 못할 확률이 높습니다.
- 따라서, 저희는 토큰을 전부 확인하고, 수작업으로 토크나이저에 토큰을 추가했습니다.
## 3.2. 비교
- Text Classification에서 좋은 성능을 보여주고 있는 다양한 NLP 모델을 활용했고, 이를 `unsmile` 테스트 데이터셋을 활용해 성능을 측정하였습니다. 성능 지표는 `sklearn`의 `roc_auc_score`를 사용하였습니다. 

- 먼저, `unsmile` 데이터셋을 활용한 여러 모델들의 결과입니다.

|모델명|clean|지역|종교|인종국적|연령|악플욕설|성소수자|성별|기타혐오|개인지칭|AVG|
|---|---|---|---|---|---|---|---|---|---|---|---|
|funnel-kor-base|0.881|0.965|0.95|0.918|0.949|0.808|0.919|0.933|0.764|0.934|0.9|
|koelectra-small-v2|0.846|0.955|0.96|0.93|0.939|0.78|0.941|0.931|0.789|0.872|0.89|
|klue-roberta-base|0.9|**0.974**|**0.988**|0.925|**0.961**|**0.843**|0.96|**0.961**|**0.886**|0.917|0.93|
|electra-kor-base|**0.905**|0.97|0.975|0.947|0.95|0.836|**0.969**|0.96|0.837|**0.956**|**0.93**|
|kobert|0.87|0.97|0.98|**0.95**|0.92|0.79|0.96|0.95|0.81|0.86|0.91|


- 여기서 성능이 좋았던 모델들에게 pseudo-labeled 된 `apeach` 데이터를 활용해 추가적으로 학습을 진행시켰고, 아래 표는 그 결과입니다.

|모델명|clean|지역|종교|인종국적|연령|악플욕설|성소수자|성별|기타혐오|개인지칭|AVG|
|---|---|---|---|---|---|---|---|---|---|---|---|
|klue-roberta-base|**0.876**|0.97|-|**0.942**|0.97|**0.82**|0.957|0.955|0.797|0.957|0.916|
|koelectra-small-v2|0.827|0.956|-|0.934|0.95|0.766|0.941|0.926|0.779|0.89|0.89|
|electra-kor-base|0.874|**0.97**|-|0.94|**0.97**|0.815|**0.968**|**0.96**|**0.946**|**0.966**|**0.934**|


- 가장 성능이 좋았던 `electra-kor-base` 모델에 추가적인 토큰을 더해서 최종 모델을 만들었고, 이를 활용해 디스코드 봇을 만들었습니다.

|모델명|clean|지역|종교|인종국적|연령|악플욕설|성소수자|성별|기타혐오|개인지칭|AVG|
|---|---|---|---|---|---|---|---|---|---|---|---|
|electra-kor-base|0.902|0.982|0.979|0.951|0.937|0.826|0.963|0.966|0.975|0.954|0.938|

# 4. 시연

![시연영상](https://user-images.githubusercontent.com/42201371/166636747-3411db33-48e8-42f7-9a6d-570ea45b296f.gif)

# 5. reference

[funnel](https://github.com/kiyoungkim1/LMkor)
[koelectra](https://github.com/kiyoungkim1/LMkor)
[roberta](https://huggingface.co/klue/roberta-base)
[kobert](https://github.com/SKTBrain/KoBERT)
[koeran unsmile datsaet](https://github.com/smilegate-ai/korean_unsmile_dataset?fbclid=IwAR0xTlHYCWK0LtrghSL1bPm2su69-LbjisutmcvLlERlHzroMlVpHq3h71g)
[apeach](https://github.com/jason9693/APEACH?fbclid=IwAR2ZBPFnv8qSy1RRqISoGkTfqmitoSLz0Fma3iPv4PZJvkZo5lAm9kForo8)
[earlystopping](https://github.com/Bjarten/early-stopping-pytorch)
[meta pseudo label](https://arxiv.org/abs/2003.10580)
