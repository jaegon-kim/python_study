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

### Numpy Deep Learning

https://toyourlight.tistory.com/74 ()

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


## Goal 2 - Natural Language Processing with Deep Learning
https://wikidocs.net/book/2155

Vector, Matrix, Tensor 
* https://wikidocs.net/52460


## Metaplot library Usage References
https://jehyunlee.github.io/2021/07/10/Python-DS-80-mpl3d2/

## 허깅 페이스
https://huggingface.co/ (sorryvolki)
 * https://huggingface.co/jaegon-kim/distilbert-base-uncased-finetuned-emotion

## 허깅 페이스로 한국어 번역기 만들기 Tutorial
https://metamath1.github.io/blog/posts/gentle-t5-trans/gentle_t5_trans.html?utm_source=pytorchkr

## DistilBERT (Distilled BERT, 지식 증류 기반 BERT)
 - 교사 BERT를 이용해서 학생 BERT를 학습 시킴
https://ariz1623.tistory.com/312
