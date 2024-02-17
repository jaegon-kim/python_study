import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def learn_logic_gateway(title, input , target):
    model = nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid()
    )

    print('\nlogic_gw: ', title)
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

torch.set_printoptions(sci_mode=False, precision=5)

input_d = torch.tensor([[0., 0.],
                      [1., 0.],
                      [0., 1.],
                      [1., 1.]])

target_or = torch.tensor([[0.],
                          [1.],
                          [1.],
                          [1.]])

learn_logic_gateway('OR', input_d, target_or)


target_and = torch.tensor([[0.],
                          [0.],
                          [0.],
                          [1.]])

learn_logic_gateway('AND', input_d, target_and)


target_nand = torch.tensor([[1.],
                          [1.],
                          [1.],
                          [0.]])

learn_logic_gateway('NAND', input_d, target_nand)


target_xor = torch.tensor([[0.],
                          [1.],
                          [1.],
                          [0.]])

learn_logic_gateway('XOR', input_d, target_nand)

