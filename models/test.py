import torch
import torch.nn.functional as F

x = torch.load('mse_tensorX.pt')
y = torch.load('mse_tensorY.pt')
# print(x)
# print(torch.sort(torch.flatten(x))[0:10], torch.sort(torch.flatten(x))[-10:])
print(torch.sort(torch.flatten(y))[0:4])
print("============")
print(torch.sort(torch.flatten(y))[-4:])
print("============")
# print(torch.sort(torch.flatten(torch.nan_to_num(y, nan=0.0))))

eps = 1e-6
criterion = F.mse_loss
loss = torch.sqrt(criterion(x, y) + eps)
print(loss)
