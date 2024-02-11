import torch

def test_cuda_available(): 
    print("cuda" if torch.cuda.is_available() else "cpu")

def test_long_tensor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = torch.LongTensor([[1, 3, 4, 5, 6,    8, 7, 2, 9, 7]]).to(device)
    print(src, src.shape)

    data = [[1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]]
    src = torch.LongTensor(data).to(device)
    print(src, src.shape)

def test_ones():
    src_mask = torch.ones(1, 1, 10)
    print(src_mask, src_mask.shape)

test_cuda_available()
test_long_tensor()
test_ones()
