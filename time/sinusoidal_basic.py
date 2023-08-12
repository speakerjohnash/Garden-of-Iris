import torch
from datetime import datetime
from calendar import monthrange
from math import sin, cos, pi

import pandas as pd
import numpy as np

class SinusoidalBasicEncoder:

    def encode(self, start_date, claim_date):

        if isinstance(start_date, np.datetime64):
            start_date = pd.Timestamp(start_date).to_pydatetime()
        if isinstance(claim_date, np.datetime64):
            claim_date = pd.Timestamp(claim_date).to_pydatetime()

        days_in_month = monthrange(claim_date.year, claim_date.month)[1]

        # Extracting the year difference
        year_difference = claim_date.year - start_date.year

        # Extracting date components
        minute = claim_date.minute
        hour = claim_date.hour 
        day = claim_date.day
        month = claim_date.month

        # Calculations
        minute_enc = [sin(2 * pi * minute / 60), cos(2 * pi * minute / 60)]
        hour_enc = [sin(2 * pi * hour / 24), cos(2 * pi * hour / 24)]
        day_enc = [sin(2 * pi * day / days_in_month), cos(2 * pi * day / days_in_month)]
        month_enc = [sin(2 * pi * month / 12), cos(2 * pi * month / 12)]
        year_enc = [sin(2 * pi * year_difference / 1000), cos(2 * pi * year_difference / 1000)]

        # Stacked encoding
        stacked_enc = [
            minute_enc,
            hour_enc,
            day_enc,
            month_enc,
            year_enc,
        ]
        
        return torch.tensor(stacked_enc)

if __name__ == '__main__':

    encoder = SinusoidalBasicEncoder()
    start_date = datetime(1776, 7, 4)
    claim_date = datetime(1863, 8, 19, 11, 2)
    encoded = encoder.encode(start_date, claim_date)

    print("Stacked encoding:")

    for encoding in encoded:
        print(encoding.tolist())
