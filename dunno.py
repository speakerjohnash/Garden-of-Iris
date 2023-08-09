from math import sin, cos, pi
import numpy as np
from datetime import datetime

def encode_time(start_date, claim_date, frequencies=[(1,)] * 5):
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
    for freq_tuple, value in zip(frequencies, time_components):
        encoding_for_component = [[sin(2 * pi * f * value), cos(2 * pi * f * value)] for f in freq_tuple]
        enc.append(encoding_for_component)

    return enc

start_date = datetime(1776, 7, 4)
claim_date = datetime(1863, 8, 19, 11, 2)

def print_encoding_with_dimensions(name, encoding):
    encoding_array = np.array(encoding)
    print(f"{name}:")
    print(f"Dimensions: {encoding_array.shape}")
    for component in encoding_array:
        print(component)

# Basic Encoding
basic_enc = encode_time(start_date, claim_date)
print_encoding_with_dimensions("Basic encoding", basic_enc)

# Multi-Frequency Encoding
multi_freq_frequencies = [(1, 2, 4, 8, 16)] * 5
multi_freq_enc = encode_time(start_date, claim_date, multi_freq_frequencies)
print_encoding_with_dimensions("Multi-frequency encoding", multi_freq_enc)

# Minute-Focused Encoding
minute_focus_frequencies = [(8, 16, 32, 64, 128)] + [(1, 2, 4, 8, 16)] * 4
minute_focus_enc = encode_time(start_date, claim_date, minute_focus_frequencies)
print_encoding_with_dimensions("Minute-focused encoding", minute_focus_enc)

# Normalized Frequencies
normalized_frequencies = [(1, 24, 365.25 / 12, 12, 1)] * 5
normalized_enc = encode_time(start_date, claim_date, normalized_frequencies)
print_encoding_with_dimensions("Normalized encoding", normalized_enc)

# Three-Frequency Encoding
three_freq_frequencies = [(1, 2, 4)] * 5
three_freq_enc = encode_time(start_date, claim_date, three_freq_frequencies)
print_encoding_with_dimensions("3 Frequency Encoding", three_freq_enc)