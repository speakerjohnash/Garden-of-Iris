# time2vec.py

import torch
from torch import nn 
import numpy as np
import math

class Time2VecEncoder(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = SineActivation(in_features, out_features)

    def forward(self, x):
        return self.model(x)
        
def t2v(tau, f, out_features, w, b, w0, b0):

    # k-1 periodic features
    v1 = f(torch.matmul(w, tau.T) + b)  

    # One Non-periodic feature
    v2 = torch.matmul(w0, tau.T) + b0

    return torch.cat([v1, v2], 0).T

class SineActivation(nn.Module):

    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(1, in_features))
        self.b0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.w = nn.parameter.Parameter(torch.randn(out_features-1, in_features))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1, 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

if __name__ == "__main__":

    # Usage example
    encoder = Time2VecEncoder(10, 6)
  
    tau = torch.rand(32, 10)  
    out = encoder(tau)
  
    print(out.shape)