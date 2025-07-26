import torch
seq = torch.randint(0, 5, (1, 196_608))
print(seq.shape, "shape of seq")