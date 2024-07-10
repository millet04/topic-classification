# 1. Classification

## Introduction
- Classifier를 이용해 사전학습모델(PLM)을 Fine-tuning 하는 방법입니다.
- 일반적으로 Classification 테스크에서 사용되는 방법입니다. 

<img src="../images/classification_2.PNG" alt="example image" width="400" height="200"/>

&nbsp;&nbsp;&nbsp;&nbsp; (블로그: https://snumin44.tistory.com/13)


## Experiment

- 모델, 데이터셋 등 구체적인 실험 정보는 다음과 같습니다.
  
  - PLM: klue/bert-base (한국어 BERT)
  - Dataset: KLUE 'ynat' (7 classes : IT/과학, 경제, 사회, 생활문화, 세계, 스포츠, 정치 )
  - Epochs: 10
  - Early Stop: 5
  - Batch Size: 128
  - Max Length: 256
  - Optimizer: AdamW
  - Learning Rate: 5e-5

- Pooler 에 따른 성능(accuracy)을 비교하면 다음과 같습니다.

||pooler_output|cls|mean|max|
|:---:|:---:|:---:|:---:|:---:|
|ynat (valid set)|86.03 (%)|86.12 (%)|85.81 (%)|86.20 (%)|

- 직접 구축한 소규모 데이터 셋(175개 문장)으로 평가한 결과는 다음과 같습니다.

||pooler_output|cls|mean|max|
|:---:|:---:|:---:|:---:|:---:|
|ynat (valid set)|78.29 (%)|80.55 (%)|78.86 (%)|78.29 (%)|

## Implementation
- 다음과 같이 직접 모델을 학습하고 평가할 수 있습니다. (bin 파일만 저장됩니다.)
```
```
- 다음과 같이 커스텀 데이터 셋(소규모 데이터 셋)을 평가할 수 있습니다.
```
``` 
