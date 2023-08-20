import torch
from datetime import datetime
from math import sin, cos, pi
import pandas as pd

class UnixTimePositionalEncoder:

    def __init__(self, d_model=32):
        self.d_model = d_model

    def encode(self, start_date, claim_date):

        # No need to convert strings to datetimes

        time_diff = (claim_date - start_date).total_seconds()
        
        # Nested cyclic encoding 
        minute_of_hour = [sin(time_diff * 2*pi / 60)] * self.d_model + [cos(time_diff * 2*pi / 60)] * self.d_model
        hour_of_day = [sin(time_diff * 2*pi / 3600)] * self.d_model + [cos(time_diff * 2*pi / 3600)] * self.d_model   
        day_of_month = [sin(time_diff * 2*pi / 86400)] * self.d_model + [cos(time_diff * 2*pi / 86400)] * self.d_model
        month_of_year = [sin(time_diff * 2*pi / 2592000)] * self.d_model + [cos(time_diff * 2*pi / 2592000)] * self.d_model
        year_of_1000 = [sin(time_diff * 2*pi / 31536000)] * self.d_model + [cos(time_diff * 2*pi / 31536000)] * self.d_model

        stacked_enc = [minute_of_hour, hour_of_day, day_of_month, month_of_year, year_of_1000]

        return torch.tensor(stacked_enc)

if __name__ == '__main__':

    encoder = UnixTimePositionalEncoder(d_model=32)

    start_date = datetime(2023, 1, 1)
    claim_date = datetime(2023, 2, 1)

    encoded = encoder.encode(start_date, claim_date)

    print("Unix time positional encoding:")
    for encoding in encoded:
        print(encoding.tolist())