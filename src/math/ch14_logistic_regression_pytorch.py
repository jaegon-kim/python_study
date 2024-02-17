import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from numpy import genfromtxt 

data = genfromtxt('NBA_Rookie_draft.csv', delimiter=',', skip_header = 1)
input = data[:, 0:3]
target = data[:, -1:]

input = torch.FloatTensor(input)
target = torch.FloatTensor(target)

model = nn.Sequential(  # 간단한 순차 모델
    nn.Linear(3, 1),    # 입력은 3개고 출력은 1인 Linear Layer이다. 
    nn.Sigmoid()        # 활성화 함수로, True/False를 적용하기 위해 sigmoid를 적용한다. 
)

print(model(input))

# SGD(Stochastic Gradient Descent) - 확률적 경사 하강법 옵티마이저
# - 전체 데이터 셋을 사용하는 대신, mini-batch를 사용하여 각 반복에서 gradient를 추정
# - 각 매개 변수에 대하여 x2 = x1 - learning_rate x gradient 를 적용하여 가중치를 업데이트 함
# - step() 메서드에 의해 매개 변수가 업데이트 된다. 
# - lr=0.001 은 Learning Rate이다. 
optimizer = optim.SGD(model.parameters(), lr=0.001)
epoches = 3000

for epoche in range(epoches + 1):
    pred = model(input)
    loss = F.binary_cross_entropy(pred, target)
    optimizer.zero_grad() # mini-batch 마다 gradient를 초기화하여, mini-batch간 영향을 없앰
    loss.backward() # 역전파하고 미분이 계산
    optimizer.step() # 매개 변수가 업데이트 된다.

print(model(input))
print(list(model.parameters()))