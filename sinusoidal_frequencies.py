# sinusoidal_frequencies.py

from math import sin, cos, pi
import numpy as np
from datetime import datetime
import calendar

class SinusoidalFrequenciesEncoder:

    def encode(self, start_date, claim_date, frequencies=[(1,)] * 5):

        # Extract time components
        time_components = [
            claim_date.minute / 60.0,
            claim_date.hour / 24.0, 
            claim_date.day / 31.0,
            claim_date.month / 12.0,
            (claim_date.year - start_date.year) / 1000.0
        ]

        # Unix timestamp (linear component)
        unix_time_component = calendar.timegm(claim_date.timetuple())

        # Encoding arrays
        enc = []
        for freq_tuple, value in zip(frequencies, time_components):
            encoding_for_component = [[sin(2 * pi * f * value), cos(2 * pi * f * value)] for f in freq_tuple]
            enc.append(encoding_for_component)

        return enc

    def print_encoding(self, name, encoded):

        sub_labels = [
            "Minute of Hour",
            "Hour of Day", 
            "Day of Month",
            "Month of Year",
            "Year of Millennium"
        ]
        
        encoding_array = np.array(encoded)
        
        print(f"\n{name}:") 
        print(f"Dimensions: {encoding_array.shape}")
        print()

        for idx, component in enumerate(encoding_array):
            print(f"{sub_labels[idx]}:")
            print(component)
            print()

if __name__ == '__main__':

  encoder = SinusoidalFrequenciesEncoder()
  
  start_date = datetime(1776, 7, 4)
  claim_date = datetime(1863, 8, 19, 11, 2)

  # Encodings
  basic_enc = encoder.encode(start_date, claim_date)
  encoder.print_encoding("Basic Encoding", basic_enc)

  multi_freq_enc = encoder.encode(start_date, claim_date, [(1,2,4,8,16)]*5)
  encoder.print_encoding("Multi-Freq Encoding", multi_freq_enc)

  # Minute-Focused Encoding
  minute_focus_frequencies = [(8, 16, 32, 64, 128)] + [(1, 2, 4, 8, 16)] * 4
  minute_focus_enc = encoder.encode(start_date, claim_date, minute_focus_frequencies)
  encoder.print_encoding("Minute-focused encoding", minute_focus_enc)

  # Three-Frequency Encoding
  three_freq_frequencies = [(1, 2, 4)] * 5
  three_freq_enc = encoder.encode(start_date, claim_date, three_freq_frequencies)
  encoder.print_encoding("3 Frequency Encoding", three_freq_enc)