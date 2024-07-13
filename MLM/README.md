# 2. Masked Language Modeling (MLM)

## Introduction

- MLM Head를 이용해 프롬프트의 [MASK] 토큰을 예측하는 방법입니다.
- Verbalizer로 예측한 토큰과 레이블을 연결해 MLM 테스크를 분류 테스크로 전환합니다. [BLOG](https://snumin44.tistory.com/15)
- 참고: [Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://github.com/timoschick/pet)
- __License__ :This project is licensed under the Apache License, Version 2.0. See the LICENSE file for details.   

<img src="../images/petmlm2.PNG" alt="example image" width="420" height="220"/>

      
## Experiment

- 모델, 데이터셋 등 구체적인 실험 정보는 다음과 같습니다.
  
  - PLM: __klue/bert-base__ (한국어 BERT)
  - Dataset: __KLUE 'ynat'__ (7 classes : IT/과학, 경제, 사회, 생활문화, 세계, 스포츠, 정치 )
  - Verbalizer:  IT/과학(0) : '과학', 경제(1): '경제', 사회(2): '사회', 생활문화(3): '생활', 세계(4): '세계', 스포츠(5): '스포츠', 정치(6): '정치'
  - Epochs: 10
  - Early Stop: 5
  - Batch Size: 128
  - Max Length: 256
  - Optimizer: AdamW
  - Learning Rate: 5e-5

- 패턴에 따른 성능(accuracy)을 비교하면 다음과 같습니다.

||[MASK]:{}|{}의 주제:[MASK]|{}의 주제는 [MASK]이다. |토픽은 무엇일까?[MASK]:{}|
|:---:|:---:|:---:|:---:|:---:|
|ynat (valid set)|87.22 (%)|87.17 (%)|87.19 (%))|87.12 (%)|

- 직접 구축한 소규모 데이터 셋(175개 문장)으로 평가한 결과는 다음과 같습니다.

||[MASK]:{}|{}의 주제:[MASK]|{}의 주제는 [MASK]이다. |토픽은 무엇일까?[MASK]:{}|
|:---:|:---:|:---:|:---:|:---:|
|커스텀 데이터 셋|78.81 (%)|78.79 (%)|78.24 (%)|81.1 (%)|

## Implementation
- 다음과 같이 직접 모델을 학습하고 평가할 수 있습니다. (bin 파일만 저장됩니다.)
```
git clone https://github.com/snumin44/topic-classification.git
cd MLM/train
sh run_train.sh
```
- 다음과 같이 커스텀 데이터 셋(소규모 데이터 셋)을 평가할 수 있습니다.
```
cd MLM/evaluation
sh run_evaluate.sh
``` 
## Citing

```
@article{schick2020exploiting,
  title={Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference},
  author={Timo Schick and Hinrich Schütze},
  journal={Computing Research Repository},
  volume={arXiv:2001.07676},
  url={http://arxiv.org/abs/2001.07676},
  year={2020}
}
@article{schick2020small,
  title={It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners},
  author={Timo Schick and Hinrich Schütze},
  journal={Computing Research Repository},
  volume={arXiv:2009.07118},
  url={http://arxiv.org/abs/2009.07118},
  year={2020}
}
```
