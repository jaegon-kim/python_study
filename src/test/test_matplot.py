import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
import matplotlib.pyplot as plt



def display_tensor():
    #'''
    tensor = torch.FloatTensor([
                                [0.1, 0.2, 0.3],
                                [0.4, 0.5, 0.6],
                                [0.7, 0.8, 0.9]
                        ])
    #'''
    #tensor = torch.randn(3, 3)
    print(tensor, '\n', tensor.shape)

    # Matplotlib를 사용하여 텐서를 이미지로 그리기
    plt.imshow(tensor, aspect='auto', cmap='gray')
    plt.colorbar()  # 컬러바 추가
    plt.show()

torch.set_printoptions(sci_mode=False, precision=5)

display_tensor()