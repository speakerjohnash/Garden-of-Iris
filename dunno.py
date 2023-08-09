from math import sin, cos, pi
from datetime import datetime

def encode_time(start_date, claim_date, frequencies=[(1, 1, 1, 1, 1)]):
    # Extract time components
    time_components = [
        claim_date.minute / 60.0,
        claim_date.hour / 24.0,
        claim_date.day / 31.0,
        claim_date.month / 12.0,
        (claim_date.year - start_date.year) / 1000.0
    ]

    # Encoding arrays
    enc = []
    for freq_tuple in frequencies:
        encoding = []
        for f, value in zip(freq_tuple, time_components):
            encoding += [sin(2 * pi * f * value), cos(2 * pi * f * value)]
        enc.append(encoding)

    return enc

start_date = datetime(1776, 7, 4)
claim_date = datetime(1863, 8, 19, 11, 2)

# Basic Encoding
basic_enc = encode_time(start_date, claim_date)
print("Basic encoding:")
print(basic_enc)

# Multi-Frequency Encoding
multi_freq_frequencies = [(1, 2, 4, 8, 16)] * 5
multi_freq_enc = encode_time(start_date, claim_date, multi_freq_frequencies)
print("Multi-frequency encoding:")
print(multi_freq_enc)

# Minute-Focused Encoding
minute_focus_frequencies = [(8, 16, 32, 64, 128)] + [(1, 2, 4, 8, 16)] * 4
minute_focus_enc = encode_time(start_date, claim_date, minute_focus_frequencies)
print("Minute-focused encoding:")
print(minute_focus_enc)
