import torch








if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3, 4], [6, 7, 8, 9]])
    x = torch.unsqueeze(x, dim=0)
    x = x.view(1, -1, 2)
    print(x)
    print(x.shape)




