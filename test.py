import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


a = torch.ones(5)#.to(device)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)
