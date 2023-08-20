import torch
from datetime import datetime
from math import sin, cos, pi
import pandas as pd

class UnixTimePositionalEncoder:

    def encode(self, start_date, claim_date):

        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date).to_pydatetime()
        if isinstance(claim_date, str):
            claim_date = pd.Timestamp(claim_date).to_pydatetime()

        # Calculate Unix time difference in seconds
        unix_difference = (claim_date - start_date).total_seconds()

        # Calculate frequency scaling factors for different time components
        scaling_factors = [60, 3600, 86400, 2592000, 31536000] # Corresponding to minute, hour, day, month, year

        # Stacked encoding
        stacked_enc = []
        for scale in scaling_factors:
            scaled_unix = unix_difference / scale
            component_enc = [sin(2 * pi * scaled_unix), cos(2 * pi * scaled_unix)]
            stacked_enc.append(component_enc)

        return torch.tensor(stacked_enc)

if __name__ == '__main__':

    encoder = UnixTimePositionalEncoder()
    start_date = '2023-01-01'
    claim_date = '2023-02-01'
    encoded = encoder.encode(start_date, claim_date)

    print("Unix time positional encoding:")
    for encoding in encoded:
        print(encoding.tolist())
