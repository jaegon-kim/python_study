# python_study

## Goal 1 - Deep learning & Mathmatics

https://toyourlight.tistory.com/category/%ED%8C%8C%EC%9D%B4%EC%8D%AC%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D/%EB%94%A5%EB%9F%AC%EB%8B%9D%EA%B3%BC%20%EC%88%98%ED%95%99

### 함수와 합성 합수
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


https://toyourlight.tistory.com/category/%ED%8C%8C%EC%9D%B4%EC%8D%AC%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D/Numpy%20%EB%94%A5%EB%9F%AC%EB%8B%9D?page=12
next:https://toyourlight.tistory.com/11


## Goal 2 - Natural Language Processing with Deep Learning
https://wikidocs.net/book/2155


## Metaplot library Usage References
https://jehyunlee.github.io/2021/07/10/Python-DS-80-mpl3d2/