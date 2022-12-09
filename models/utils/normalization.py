import torch


class Normalizer(torch.nn.Module):
    def __init__(self, means, stdevs):
        super().__init__()

        self.mean = torch.Tensor(means)
        self.std = torch.Tensor(stdevs)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x

    def revert(self, x):
        x = x * self.std + self.mean
        return x