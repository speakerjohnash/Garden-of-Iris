import torch
from datetime import datetime
from math import sin, cos, pi
import pandas as pd

from calendar import monthrange

class UnixTimePositionalEncoder:

    def __init__(self, d_model=1):
        self.d_model = d_model

    def encode(self, start_date, claim_date):

        # No need to convert strings to datetimes

        time_diff = (claim_date - start_date).total_seconds()

        # Get the number of days in the claim date's month
        days_in_month = monthrange(claim_date.year, claim_date.month)[1]
        
        # Nested cyclic encoding 
        cycle_of_minute = [sin(time_diff * 2*pi / 60)] * self.d_model + [cos(time_diff * 2*pi / 60)] * self.d_model
        cycle_of_hour = [sin(time_diff * 2*pi / 3600)] * self.d_model + [cos(time_diff * 2*pi / 3600)] * self.d_model   
        cycle_of_day = [sin(time_diff * 2*pi / 86400)] * self.d_model + [cos(time_diff * 2*pi / 86400)] * self.d_model
        cycle_of_month = [sin(time_diff * 2*pi / 2592000)] * self.d_model + [cos(time_diff * 2*pi / 2592000)] * self.d_model
        cycle_of_year = [sin(time_diff * 2*pi / 31536000)] * self.d_model + [cos(time_diff * 2*pi / 31536000)] * self.d_model

        stacked_enc = cycle_of_minute + cycle_of_hour + cycle_of_day + cycle_of_month + cycle_of_year

        return torch.tensor(stacked_enc)

if __name__ == '__main__':

    encoder = UnixTimePositionalEncoder(d_model=1)

    start_date = datetime(1776, 7, 4)
    claim_date = datetime(1863, 8, 19, 11, 2)

    encoded = encoder.encode(start_date, claim_date)

    print("Unix time positional encoding:")
    for encoding in encoded:
        print(encoding.tolist())
