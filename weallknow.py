import torch
from torch import nn
import numpy as np
from math import sin, cos, pi
import calendar
from datetime import datetime

class TimeEncodingModule(nn.Module):
    def __init__(self, in_features, out_features, frequencies=[(1,)] * 5):
        super(TimeEncodingModule, self).__init__()
        self.out_features = out_features
        self.frequencies = frequencies
        self.w0 = nn.parameter.Parameter(torch.randn(1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def encode_time(self, time_components, unix_time_component):
        enc = []
        for freq_tuple, value in zip(self.frequencies, time_components):
            encoding_for_component = [[sin(2 * pi * f * value), cos(2 * pi * f * value)] for f in freq_tuple]
            enc.append(encoding_for_component)

        # Converting the encoded time to a tensor
        enc_tensor = torch.tensor(enc, dtype=torch.float32).view(-1, len(time_components) * 2)

        # Combining linear and periodic components
        combined_enc = self.f(torch.matmul(enc_tensor, self.w) + self.b)

        # Adjusting the computation of linear_enc
        linear_enc = (unix_time_component.view(-1, 1) * self.w0) + self.b0  # Reshaping to a 2D tensor
        linear_enc = linear_enc.expand(combined_enc.size(0), -1)  # Expanding to match the shape of combined_enc

        return torch.cat([combined_enc, linear_enc], 1)  # Concatenating along the correct dimension

    def forward(self, start_date, claim_date):

        time_components = [
            claim_date.minute / 60.0,
            claim_date.hour / 24.0,
            claim_date.day / 31.0,
            claim_date.month / 12.0,
            (claim_date.year - start_date.year) / 1000.0
        ]

        unix_time_component = unix_timestamp(claim_date)

        return self.encode_time(time_components, unix_time_component)

def unix_timestamp(dt):
    return torch.tensor([calendar.timegm(dt.timetuple())], dtype=torch.float32)  # Returning as a 1D tensor

# Example usage:
start_date = datetime(2021, 1, 1)
claim_date = datetime.now()
time_enc_module = TimeEncodingModule(10, 6)
embedding = time_enc_module(start_date, claim_date)
print(embedding)
