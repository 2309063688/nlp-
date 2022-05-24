import torch

a = torch.zeros(1,2).cuda()
b = torch.zeros(1,2)

if a.device != b.device:
    b = b.to(device=a.device)

print(b.device)
