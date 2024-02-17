import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def logic_gateway_single_layer(title, input , target):
    model = nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid()
    )

    print('\nlogic_gw (Single Layer): ', title)
    print('expect: \n', target)

    optimizer = optim.SGD(model.parameters(), lr=1)
    epoches = 300

    for epoche in range(epoches + 1):
        pred = model(input)
        loss = F.binary_cross_entropy(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('result: \n', model(input))



def logic_gateway_double_layer(title, input , target):
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.Sigmoid(),
        nn.Linear(2, 1),
        nn.Sigmoid()
    )

    print('\nlogic_gw (Double Layer): ', title)
    print('expect: \n', target)

    optimizer = optim.SGD(model.parameters(), lr=1)
    epoches = 2000

    for epoche in range(epoches + 1):
        pred = model(input)
        loss = F.binary_cross_entropy(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('result: \n', model(input))




torch.set_printoptions(sci_mode=False, precision=5)

input_d = torch.tensor([[0., 0.],
                      [1., 0.],
                      [0., 1.],
                      [1., 1.]])

target_or = torch.tensor([[0.],
                          [1.],
                          [1.],
                          [1.]])

target_and = torch.tensor([[0.],
                          [0.],
                          [0.],
                          [1.]])

target_nand = torch.tensor([[1.],
                          [1.],
                          [1.],
                          [0.]])

target_xor = torch.tensor([[0.],
                          [1.],
                          [1.],
                          [0.]])

logic_gateway_single_layer('OR', input_d, target_or)
logic_gateway_single_layer('AND', input_d, target_and)
logic_gateway_single_layer('NAND', input_d, target_nand)
logic_gateway_single_layer('XOR', input_d, target_xor)

logic_gateway_double_layer('XOR', input_d, target_xor)

logic_gateway_double_layer('OR', input_d, target_or)
logic_gateway_double_layer('AND', input_d, target_and)
logic_gateway_double_layer('NAND', input_d, target_nand)


