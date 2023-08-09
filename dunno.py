from math import sin, cos, pi 
from datetime import datetime

def encode_time(start_date, claim_date):

    # Frequencies
    freqs = [1, 2, 4, 8, 16]
    
    # Extract time components
    minute = claim_date.minute / 60.0  
    hour = claim_date.hour / 24.0
    day = claim_date.day / 31.0
    month = claim_date.month / 12.0
    year = (claim_date.year - start_date.year) / 1000.0

    # Encoding arrays
    enc = []
    for f in freqs:
        minute_enc = [sin(2*pi*f*minute), cos(2*pi*f*minute)]
        hour_enc = [sin(2*pi*f*hour), cos(2*pi*f*hour)]
        day_enc = [sin(2*pi*f*day), cos(2*pi*f*day)]
        month_enc = [sin(2*pi*f*month), cos(2*pi*f*month)]
        year_enc = [sin(2*pi*f*year), cos(2*pi*f*year)]
        
        # Append all frequency encodings 
        enc.append(minute_enc + hour_enc + day_enc + month_enc + year_enc)

    return enc

start_date = datetime(1776, 7, 4)
claim_date = datetime(1863, 8, 19, 11, 2)

enc = encode_time(start_date, claim_date)
print(enc)