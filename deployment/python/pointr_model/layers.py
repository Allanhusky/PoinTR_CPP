import torch
from torch import nn
from torch.nn.init import trunc_normal_

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob=drop_prob

    def forward(self,x):
        if self.training and self.drop_prob:
            raise RuntimeError("This model copy supports inference only")
        return x

def print_log(message, **kwargs):
    print(message)
