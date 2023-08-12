from math import sin, cos, pi

import numpy as np
import pandas as pd

from datetime import datetime
import calendar
import torch

class SinusoidalFrequenciesEncoder:

	def __init__(self, frequencies=[(1,)] * 5):
		self.frequencies = frequencies

	def encode(self, start_date, claim_date):

		if isinstance(start_date, np.datetime64):
			start_date = pd.Timestamp(start_date).to_pydatetime()
		if isinstance(claim_date, np.datetime64):
			claim_date = pd.Timestamp(claim_date).to_pydatetime()

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
		for freq_tuple, value in zip(self.frequencies, time_components):
			encoding_for_component = [[sin(2 * pi * f * value), cos(2 * pi * f * value)] for f in freq_tuple]
			enc.append(encoding_for_component)

		return torch.tensor(enc)  # Convert to PyTorch tensor

	def print_encoding(self, name, encoded_tensor):

		encoded = encoded_tensor.numpy()  # Convert to numpy array

		sub_labels = [
			"Minute of Hour",
			"Hour of Day", 
			"Day of Month",
			"Month of Year",
			"Year of Millennium"
		]
		
		print(f"\n{name}:") 
		print(f"Dimensions: {encoded.shape}")
		print()

		for idx, component in enumerate(encoded):
			print(f"{sub_labels[idx]}:")
			print(component)
			print()

if __name__ == '__main__':

	start_date = datetime(1776, 7, 4)
	claim_date = datetime(1863, 8, 19, 11, 2)

	# Basic Encoding
	basic_encoder = SinusoidalFrequenciesEncoder()
	basic_enc = basic_encoder.encode(start_date, claim_date)
	basic_encoder.print_encoding("Basic Encoding", basic_enc)

	# Multi-Freq Encoding
	multi_freq_encoder = SinusoidalFrequenciesEncoder([(1,2,4,8,16)]*5)
	multi_freq_enc = multi_freq_encoder.encode(start_date, claim_date)
	multi_freq_encoder.print_encoding("Multi-Freq Encoding", multi_freq_enc)

	# Minute-Focused Encoding
	minute_focus_frequencies = [(8, 16, 32, 64, 128)] + [(1, 2, 4, 8, 16)] * 4
	minute_focus_encoder = SinusoidalFrequenciesEncoder(minute_focus_frequencies)
	minute_focus_enc = minute_focus_encoder.encode(start_date, claim_date)
	minute_focus_encoder.print_encoding("Minute-focused encoding", minute_focus_enc)

	# Three-Frequency Encoding
	three_freq_frequencies = [(1, 2, 4)] * 5
	three_freq_encoder = SinusoidalFrequenciesEncoder(three_freq_frequencies)
	three_freq_enc = three_freq_encoder.encode(start_date, claim_date)
	three_freq_encoder.print_encoding("3 Frequency Encoding", three_freq_enc)
