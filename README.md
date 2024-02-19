# python_study

## Goal 1 - Deep learning & Mathmatics

https://toyourlight.tistory.com/category/%ED%8C%8C%EC%9D%B4%EC%8D%AC%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D/%EB%94%A5%EB%9F%AC%EB%8B%9D%EA%B3%BC%20%EC%88%98%ED%95%99

### 기초 수학
https://toyourlight.tistory.com/4
* src/math/ch1_composite_func.py

https://toyourlight.tistory.com/5
* src/math/ch2_derive.py
* src/math/ch2_drive_2.py

https://toyourlight.tistory.com/6
* src/math/ch3_comp_func.py
* src/math/ch3_2_comp_func_derive.py

https://toyourlight.tistory.com/7
* src/math/ch4_comp_vector_input.py

https://toyourlight.tistory.com/8 (전치행렬, transpose 사용)
* src/math/ch5_comp_vector_derive.py

https://toyourlight.tistory.com/9

https://toyourlight.tistory.com/10 (2차원 행렬을 입력 받는 합성 함수의 도함수)
* src/math/ch7_comp_2xmatrix_derive.py
* src/math/ch7_pytorch.py

### 평균(average), 편차(deviation), 분산(variance), 표준편차(variance), 정규화(normalization), 표준화(standardization)
https://toyourlight.tistory.com/36
* ch36_representative_value.py
* min-max 정규화 (0~1 까지의 값으로 normalziation)
* Z score 표준화 (deviation/standard-deviation)

### Loss Function

 MSE (Mean Square Error)
  * 차분의 제곱의 평균 (.. 편차와 유사 ..?)

 Binary Cross Entropy Error (BCEE)
  * 결과 값이 Binary 형태 일때 사용하기 적절한 Loss 함수 이다.
  * 정답이 1인데, predict(W, B) = 0 이거나, 정답이 0인데, predict(W, B) = 1이면 오차가 크다.
  * 정답이 0인데, predict(W, B) = 0 이거나, 정답이 1인데, predict(W, B) = 1이면 오차가 작다.
  * '-log(x)' 와 '-log(1-x)'를 합쳐서 만든다.
  * https://toyourlight.tistory.com/14 에 BCEE에 대해 설명한다.

 Cross Entropy
 

 KL Divergence (Kullback-Leibler Divergence)



### Activation Function

 * 선형함수를 적층해 봤자 결국 선형함수가 됨. f(x) = y = WX 일때, y = f( f( f(x) ))로 적층해 봤자, y = W^3 X가 되어 선형함수가 되어 버림. Hidden Layer가 3개인 신경망과 1개인 신경망의 차이가 없다.
 * 선형 함수에 비선형 함수를 합성하여 선형성을 없앤다.
 * Ref: https://kevinitcoding.tistory.com/entry/%ED%99%9C%EC%84%B1%ED%99%94-%ED%95%A8%EC%88%98-%EC%A0%95%EC%9D%98%EC%99%80-%EC%A2%85%EB%A5%98-%EB%B9%84%EC%84%A0%ED%98%95-%ED%95%A8%EC%88%98%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%B4%EC%95%BC-%ED%95%98%EB%8A%94-%EC%9D%B4%EC%9C%A0

 sigmoid()
 * 무한대의 실수 값을 0~1사이의 확률값으로 변환한다. Binary Classification에 많이 사용된다.
 * 2차 AI 혹한기 전까지 가장 보편적으로 사용된 Activation Function이다.
 * 입력 값의 절대 값이 커질 수 록 기울기가 0으로 수렴하는 단점이 있다.
 * https://toyourlight.tistory.com/14 : 선형 함수의 결과 값에 sigmoid를 적용하여, 이진 분류 1, 0을 만들 수 있다. (농구 선수의 성적을 기반으로 draft 성공/실패 분류 등) 
                   
softmax 
 * 비선형 출력을 확률로 처리한다. (볼츠만 분포함수랑 유사)
 * Sigmoid와 달리, 다중 분류로 사용한다. (여러 target에 대한 확률이 출력된다.) Transformer에서도 출력된 vocablary의 확률 리스트를 softmax로 구하고, 이중에서 최대 값을 선택하여 '생성'에 사용한다.
 * 단점: 값이 지수함수로 변화해서 너무 커져서 overflow가 발생할 수 있다.   


 tanh(): Hyperbolic Tangent Function
 * 쌍곡선 함수. sigmoid와 비슷하개 생김. sigmoid 대비 기울기가 작아지지 않는 구간이 넓어서 양수 음수 모두 학습 효율이 뛰어나다. sigmoid의 대안으로 활용된다. 
 * 기울기가 0으로 수렴하는 구간이 존재하여, 기울기 소실 문제에 대해 자유롭지는 않다.  

 ReLu
 * 입력값이 음수이면 0, 양수이면 그대로 흘려보냄. Sigmoid의 기울기 소실 문제를 해결함. 
 * Layer를 깊게 쌓을 수 있음. ensamble 효과.
 * 입력 값이 너무 커지면 모델이 입력 값에 편향된다. (max를 6이하로 제한)
 * 음수 값이 들어오면 0이 되기 때문에 음수 값에서는 학습이 되지 않는다. (Dying ReLu, Dead Nueron)

 Leaky ReLu
 * Dying ReLu, Dead Nueron 문제 해결
 * 입력 값에 1이하의 아주 작은 값을 곱함 
 
 ELU (Exponential Linear Unit)
 * ReLu의 장법을 포함하면서 단점을 최소화
 * 음수가 들어오면 지수를 사용해 부드럽게 꺾어준다. (노이즈에 덜 민감한 효과)

### Optimizer (최적화 방법)
 
 Gradient Descent (경사 하강법)  https://heytech.tistory.com/380
 * 문제점 1) Local Minimum에 빠지기 쉽다. 
 * 문제점 2) Saddle Point 문제 : 기울기가 0이지만 극 값이 아닌 지점

 Momentum (관성) https://heytech.tistory.com/382
 * 관성을 줘서 추가로 이동하게 함. (Local Minimum에 빠지는 경우 대처 가능)

 AdaGrad (Adaptive Gradient: 적응적 기울기) https://heytech.tistory.com/383
 * Feature 별로 학습률을 Adaptive하게 조절함 (큰 기울기를 가져 학습이 많이된 변수는 학습률을 감소 시킴. )
 * 단점: 학습이 오래 진행될 수록 학습률이 0에 가까줘지기 때문에 더이상 학습이 진행되지 않을 수 있음. (학습이 진행될 때 학습이 잘 이루어져 변수 값이 업데이트 되지 않는 것인지 아니면 학습률이 0에 가까워져서 학습이 안되는 것인지 알 수 없음)

 RMSProp (Root Mean Sqaure Propagration) https://heytech.tistory.com/384
 * 이전 time step에서의 기울기를 같은 비율로 누적하지 않고, 지수 이동 평균(Exponential Moving Average, EMA)를 활용하여 기울기를 업데이트 함. 최근 time step에서의 기울기는 많이 반영하고 먼 과거의 time step에서의 기울기는 조금만 반영함
  * 효율적인 학습이 가능하고, AdaGrad 보다 학습을 오래 할 수 있음

 Adam (Adaptive Moment Estimation(Adam)) https://heytech.tistory.com/385
  * Momentum과 RMSProp의 장점을 결합한 것임. 
  * 딥러닝에서 가장 많이 사용되어 오던 최적화 기법

### Numpy Deep Learning

https://toyourlight.tistory.com/74 (목차)

https://toyourlight.tistory.com/11
 * 참고: https://heytech.tistory.com/362 (나중에 따로 볼만한 내용임)
 * 참조: http://infoso.kr/?p=2645 (∑연산 성질. 오차 제곱 합에 대한 편미분) 
 * 참고: https://mazdah.tistory.com/761
 * 참고: https://j1w2k3.tistory.com/1491,

https://toyourlight.tistory.com/12 (선형 회귀)
 * ch12_linear_regression.py
 * ch12_linear_regression_pytorch.py

https://toyourlight.tistory.com/13 (입력 특성이 2개인 선형 회귀)
 * ch13_multi_attribute.py

https://toyourlight.tistory.com/14 (로지스틱 회귀 구현하기)
 * ch14_logistic_regression.py
 * ch14_logistic_regression_pytorch.py

https://toyourlight.tistory.com/16 (softmax)

https://toyourlight.tistory.com/17 (softmax, cross entropy를 이용한 classification)

https://toyourlight.tistory.com/18 (multi classification with one data entry)
 * ch18_multi_classify_softmax.py

https://toyourlight.tistory.com/19 (multi classification with data set)
 * ch18_multi_classify_softmax2.py
 * ch18_multi_classify_softmax_torch.py

https://toyourlight.tistory.com/20 (Single Layer Perceptron)
 * Activation function: sigmoid(0/1로 변환), softmax(classification 별 확률로 변환)
 * ch20_single_layer_perceptron.py (130개 data로 single layer perceptron learning)

https://toyourlight.tistory.com/21 (Limitation of single layer perceptron - XOR problem)
 * ch21_or_gatew_single_layer_perceptron.py

https://toyourlight.tistory.com/22 (Limitation of single layer perceptron - Linear regression)
 * ch22_linear_regression_single_layer_perceptron.py
 * ch22_linear_regression_single_layer_perceptron2.py

https://toyourlight.tistory.com/23 (MLP - OXR problem 기초이론)
 * https://ynebula.tistory.com/22 (보충 설명, XOR MLP의 XY 좌표 설명)
https://toyourlight.tistory.com/24 (MLP - OXR 도함수 도출)
https://toyourlight.tistory.com/25
 * ch25_xor_gateway_mlp.py
 * ch25_xor_gateway_mlp_torch.py

https://toyourlight.tistory.com/26 (MLP - non-linear regrssion)
https://toyourlight.tistory.com/27
 * 시그모이드 도함수의 값은 원래 시그모이드의 최대 1/4 밖에 되지 않는다.
 * 활성화 함수가 시그모이드인 복잡한 MLP에서는 도함수 값이 점점 0을 향해 간다.
 * 입력에 가까운 레이어의 가중치 값은 업데이트가 잘 되지 않는, 존재감이 없어지는 일이 발생 -> 기울기 손실(Gradient Vanishing) 문제 
  * ReLU(x): Rectified Linear Unit (정류된 선형 단위)
https://toyourlight.tistory.com/27 (구현)
 * ch28_non_linear_regression_mlp.py
 * ch28_non_linear_regression_mlp_torch.py

https://toyourlight.tistory.com/29 (지수 가중 이동 평균, Exponentially Weighted Moving Average)
https://toyourlight.tistory.com/30 
 * ch30_exp_weighted_moving_avg.py

https://toyourlight.tistory.com/31 (경사 하강법의 문제점)
 * ch31_gradient_descent.py

https://toyourlight.tistory.com/32 (경사 하강법의 개선, Momentum, RMSprop)
 * ch32_momentum.py
 * ch32_RMSprop.py

https://toyourlight.tistory.com/33 (경사 하강업의 개선, Adam)
 * ch32_adam.py

https://toyourlight.tistory.com/34
 * ch34_enhance_softmax_backward.py

https://toyourlight.tistory.com/35 (Minibatch & Shuffle)
 * ch35_gradient_descent.py
 * ch35_stochastic_gradient_descent.py
 * ch35_mini_batch.py
 * ch35_mini_batch_GD_shuffle.py

https://toyourlight.tistory.com/38 (Standardiztion, Normalization)


https://toyourlight.tistory.com/48 (CNN)
ch48_cnn.py

https://toyourlight.tistory.com/66 (RNN)
 * DNN(wx + b로 불리는 완전 연길 기반 인공 신경망 모들) - 입력 특성과 목적 특성간의 '관계'를 찾는 것임. 특성 값들의 배열 순서에는 의미가 없음 (입력 데이트의 특성의 열을 바꾸어 입력해도 훈련에는 아무런 변화가 없다)

https://toyourlight.tistory.com/71 (RNN - Many to One)
 * ch66_RNN_many2one.py - 컴파일 에러 발생 (수정 필요함) 

https://toyourlight.tistory.com/73 (RNN - Many to Many)
 * ch66_RNN_many2many.py 
 * vanishing gradient 문제 해결


https://toyourlight.tistory.com/category/%ED%8C%8C%EC%9D%B4%EC%8D%AC%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D/Numpy%20%EB%94%A5%EB%9F%AC%EB%8B%9D?page=12
next:https://toyourlight.tistory.com/11

## PyTorch로 시작하는 딥러닝 입문
https://wikidocs.net/32471
 * Scalar: 0 차원, Vector: 1차원, Matrix(2D Tensor): 2차원, Tensor: 3.. 차원
 * 2D tensor: |t| = (batch size, dim)
 * 3D tensor: |t| = (batch size, lenth, dim)

1 Epoch 
 * 모든 data set을 한번 학습하면 1 Epoch이다.

Sample과 특성
 * 특성: 종속 변수 y를 예측하기 위한 각각의 독립 변수 x (x1, x2, x3, x4 ....) 
 * Sample: 셀수 있는 데이터 단위 구분 {x1, x2, x3 } --> Sample

next : https://wikidocs.net/57805

## Transformer 
 https://metamath1.github.io/2021/11/11/transformer.html
 https://colab.research.google.com/github/metamath1/ml-simple-works/blob/master/transformer/annotated_transformer.ipynb

## Goal 2 - Natural Language Processing with Deep Learning
https://wikidocs.net/book/2155

Vector, Matrix, Tensor 
* https://wikidocs.net/52460


## Metaplot library Usage References
https://jehyunlee.github.io/2021/07/10/Python-DS-80-mpl3d2/

## Transformer 
 * 진짜 주석달린 Transformer https://metamath1.github.io/2021/11/11/transformer.html
 * The Illustrated Transformer https://nlpinkorean.github.io/illustrated-transformer/

## dot product similarity
https://www.aitimes.kr/news/articleView.html?idxno=17314

## 허깅 페이스
https://huggingface.co/ (sorryvolki)
 * https://huggingface.co/jaegon-kim/distilbert-base-uncased-finetuned-emotion

## 허깅 페이스로 한국어 번역기 만들기 Tutorial
https://metamath1.github.io/blog/posts/gentle-t5-trans/gentle_t5_trans.html?utm_source=pytorchkr

## BERT
 - MLM(Masked Language Model) 학습: BERT는 사전 훈련된 언어 모델로, 대량의 텍스트 데이터를 사용하여 사전에 학습됩니다. 이 모델은 주어진 문장에 빈칸을 만들고 해당 빈칸에 알맞는 단어를 맞춘다.
 - BERT는 MLM 이외에도 NSP(Next Sentence Prediction)라고 불리는 다음 문장 예측 작업을 함께 사용합니다. NSP 작업은 주어진 두 문장을 모델에 입력으로 주고, 모델은 이 두 문장이 이어진 문장인지 아니면 서로 무관한 문장인지를 예측하는 작업이다.

## DistilBERT (Distilled BERT, 지식 증류 기반 BERT)
 - 교사 BERT를 이용해서 학생 BERT를 학습 시킴
 - 
https://ariz1623.tistory.com/312

## minGPT (Andrej Kapathy)
* https://github.com/karpathy/minGPT
* Let’s build GPT - GPT를 밑바닥부터 만들어 봅시다 (https://discuss.pytorch.kr/t/geeknews-lets-build-gpt-gpt/958)

## sLLM 모델 
* 메타의 ‘라마(Llama)-2 - 알파카(Alpaca)’
* 제미나이 나노
* 파이(Phi-2)-2
* 미스트랄 7B

국내
*Huggiung Face: MoMo-70B, SOLAR-10.7B)
*

## RAG(Retrieval Augmented Generation)

## Copilot chat
* https://docs.github.com/en/copilot/github-copilot-chat/about-github-copilot-chat
* 코드의 버그 수정 가능
