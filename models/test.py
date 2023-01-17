import torch

def print_cuda_count():
    print(torch.cuda.device_count())

if __name__ == '__main__':
    print_cuda_count()