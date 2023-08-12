import torch
from torch import nn
from math import sin, cos, pi
import calendar
from datetime import datetime
from calendar import monthrange

import numpy as np
import pandas as pd

class IrisTimeEncoder(nn.Module):

    def __init__(self, in_features, out_features, frequencies=[(1,)]*5):
        super(IrisTimeEncoder, self).__init__()
        self.out_features = out_features
        self.frequencies = frequencies
        self.w0 = nn.parameter.Parameter(torch.randn(1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.f = torch.sin

    def encode_time(self, time_components, unix_time_component):
        enc = []
        for freq_tuple, value in zip(self.frequencies, time_components):
            encoding_for_component = [[sin(2 * pi * f * value), cos(2 * pi * f * value)] for f in freq_tuple]
            enc.append(encoding_for_component)
        
        enc_tensor = torch.tensor(enc, dtype=torch.float32).view(-1, len(time_components) * 2)
        combined_enc = self.f(torch.matmul(enc_tensor, self.w) + self.b)
        linear_enc = (unix_time_component.view(-1, 1) * self.w0) + self.b0
        linear_enc = linear_enc.expand(combined_enc.size(0), -1)
                
        return torch.cat([combined_enc, linear_enc], 1)

    def forward(self, start_date, claim_date):

        if isinstance(start_date, np.datetime64):
            start_date = pd.Timestamp(start_date).to_pydatetime()
        if isinstance(claim_date, np.datetime64):
            claim_date = pd.Timestamp(claim_date).to_pydatetime()

        days_in_month = monthrange(claim_date.year, claim_date.month)[1]

        time_components = [
            claim_date.minute / 60.0,
            claim_date.hour / 24.0,
            claim_date.day / days_in_month,
            claim_date.month / 12.0,
            (claim_date.year - start_date.year) / 1000.0
        ]

        unix_time_component = torch.tensor([calendar.timegm(claim_date.timetuple())], dtype=torch.float32)

        return self.encode_time(time_components, unix_time_component)

    def encode(self, start_date, claim_date):
        return self.forward(start_date, claim_date)

if __name__ == '__main__':

    start_date = datetime(1776, 7, 4)
    claim_date = datetime(1863, 8, 19, 11, 2)

    encoder = IrisTimeEncoder(10, 6)
    embedding = encoder(start_date, claim_date)

    print(embedding)