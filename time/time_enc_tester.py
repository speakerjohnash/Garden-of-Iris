import sys
import torch
import math
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  
from torch.optim import Adam
from datetime import datetime

from time2vec import Time2VecEncoder  
from time_transformer import TimeEncodingTransformer
from pos_unix_enc import  UnixTimePositionalEncoder
import sinusoidal_basic as basic
import sinusoidal_frequencies as freq
import mixed_time_enc as iris

class NullEncoder:
	def encode(self, start_date, date):
		return torch.tensor([])

# Function to encode data
def encode_data(encoder, start_date, train_dates, test_dates):
	X_train_encoded = torch.stack([encoder.encode(date, start_date) for date in train_dates])
	X_test_encoded = torch.stack([encoder.encode(date, start_date) for date in test_dates])
	return X_train_encoded, X_test_encoded

# Dictionary mapping encoder names to objects  
encoder_objects = {
	'Time2Vec': Time2VecEncoder(in_features=1, out_features=6),
	'SinusoidalBasic': basic.SinusoidalBasicEncoder(),
	'SinusoidalFrequencies': freq.SinusoidalFrequenciesEncoder(),
	'MixedTime': iris.IrisTimeEncoder(in_features=10, out_features=6),
	'UnixTimePositional': UnixTimePositionalEncoder(),
	'Null': NullEncoder() 
}

# List of encoders you want to include
encoders = ['UnixTimePositional', 'Null']  

# Load data
df = pd.read_csv('synthetic_data.csv')
df['Date'] = pd.to_datetime(df['Date']) 
df = df[['Date', 'Close']]
use_ordered_data = False
ensure_preceding_day = True
batch_size = 128
window_size = 8
gap = 1
X, y = [], []
i = 0

for target_idx in range(window_size, len(df) - gap):
	
	# Create a sliding window of the window_size days preceding the target date
	if use_ordered_data:
		preceding_idxs = range(target_idx - window_size, target_idx)
	else:
		all_preceding_data = range(target_idx) # All data before the target date
		if ensure_preceding_day:
			random_sample_size = window_size - 1
			preceding_idxs = np.random.choice(all_preceding_data[:-1], random_sample_size, replace=False)
			preceding_idxs = np.append(preceding_idxs, target_idx - 1)
		else:
			preceding_idxs = np.random.choice(all_preceding_data, window_size, replace=False)
		np.random.shuffle(preceding_idxs)  # Shuffle to randomize the position of the preceding day

	# Print the preceding dates
	preceding_dates = df.loc[preceding_idxs, 'Date'].values
	target_date = df.loc[target_idx + gap, 'Date']
	preceding_closes = df.loc[preceding_idxs, 'Close'].values
	target_close = df.loc[target_idx + gap, 'Close']

	if i == 1 or i == 9000:

		print("Context window dates:")
		print(preceding_dates)
		print("Target date:")  
		print(target_date)

	X.append(preceding_closes)
	y.append(target_close)

	i = i + 1

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
train_idxs, test_idxs = train_test_split(range(window_size, len(df) - gap), test_size=0.2, shuffle=False)
start_date = df.loc[window_size-1, 'Date']
X_train_dates = [df.loc[idx + gap, 'Date'] for idx in train_idxs] 
X_test_dates = [df.loc[idx + gap, 'Date'] for idx in test_idxs]

# Encode timestamps using different techniques
encoder_data = {}
for encoder_name in encoders:
	encoder = encoder_objects[encoder_name] 
	X_train_encoded, X_test_encoded = encode_data(encoder, start_date, X_train_dates, X_test_dates)
	encoder_data[encoder_name] = (X_train_encoded, X_test_encoded)

for encoder_name, (X_train_encoded, X_test_encoded) in encoder_data.items():

	print(f"Results for {encoder_name}:")
	
	size = X_train_encoded.shape[-1]

	print(f"Encoding shape: {X_train_encoded.shape}")
	print(f"Encoding size: {size}")
	print(f"X_train shape: {X_train.shape}")

	X_train_prices = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
	X_test_prices = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

	# Adjust the shapes for concatenation
	X_train_encoded = X_train_encoded.view(*X_train_prices.shape[:-1], -1)
	X_test_encoded = X_test_encoded.view(*X_test_prices.shape[:-1], -1)

	# Concatenate the prices with their corresponding temporal encodings
	X_train_combined = torch.cat([X_train_prices, X_train_encoded], dim=-1)
	X_test_combined = torch.cat([X_test_prices, X_test_encoded], dim=-1)

	print(f"X_train_combined shape: {X_train_combined.shape}")
	print(f"X_test_combined shape: {X_test_combined.shape}")

	y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
	y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

	print(f"Original y_test size: {len(y_test_tensor)}")
	print(f"Window size: {window_size}")

	# Calculate target size that is divisible by window_size
	orig_size = len(y_test_tensor)
	target_size = int(math.ceil(orig_size / window_size) * window_size)

	print(f"Target size: {target_size}") 

	# Calculate padding amount  
	pad_amount = target_size - orig_size

	print(f"Padding amount: {pad_amount}")

	# Dynamically pad tensor
	pad = (0, 0, pad_amount, 0) 
	y_test_tensor = F.pad(y_test_tensor, pad, value=1)

	print(f"Padded tensor size: {len(y_test_tensor)}")

	y_test_tensor = F.pad(y_test_tensor, (0, pad_amount))
	y_test_tensor = y_test_tensor.reshape(-1, window_size)

	print(f"y_train_tensor shape: {y_train_tensor.shape}")
	print(f"y_test_tensor shape: {y_test_tensor.shape}")

	X_train_tensor = torch.tensor(X_train_combined, dtype=torch.float32)
	X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32)

	# Print their shapes
	print(f"X_train_tensor shape: {X_train_tensor.shape}")
	print(f"X_test_tensor shape: {X_test_tensor.shape}")

	# Batching code
	batch_size = 128
	window_size = 8

	# Print original shapes
	print(f"Original y_train_tensor shape: {y_train_tensor.shape}")
	print(f"Original X_train_tensor shape: {X_train_tensor.shape}")

	# Reshape the data
	X_train_tensor = X_train_tensor.reshape(-1, window_size, X_train_tensor.shape[-1])
	X_test_tensor = X_test_tensor.reshape(-1, window_size, X_test_tensor.shape[-1])

	num_train_batches = len(X_train_tensor) // batch_size
	num_test_batches = len(X_test_tensor) // batch_size

	X_train_tensor = X_train_tensor[:num_train_batches * batch_size]
	y_train_tensor = y_train_tensor[:num_train_batches * batch_size]  # Adjust size to match X_train_tensor

	# Reshape y_train_tensor to match X_train_tensor
	y_train_tensor = y_train_tensor.reshape(-1, window_size, 1)

	# Print shapes after reshaping
	print(f"Reshaped y_train_tensor shape: {y_train_tensor.shape}")
	print(f"Reshaped X_train_tensor shape: {X_train_tensor.shape}")

	X_test_tensor = X_test_tensor[:num_test_batches * batch_size] 
	y_test_tensor = y_test_tensor[:num_test_batches * batch_size]

	# Print batched shapes
	print(f"Batched X_train_tensor shape: {X_train_tensor.shape}")
	print(f"Batched X_test_tensor shape: {X_test_tensor.shape}")

	# Sequential model for benchmark
	model = nn.Sequential(
		nn.Linear(X_train_combined.shape[1], 128),
		nn.ReLU(),
		nn.Linear(128, 64), 
		nn.ReLU(),
		nn.Linear(64, 1)
	)

	model = TimeEncodingTransformer(input_dim=X_train_combined.shape[-1])

	optimizer = Adam(model.parameters(), lr=0.001)
	loss_fn = nn.MSELoss()

	# Train  
	model.train()
	epochs = 250

	for epoch in range(epochs):
		optimizer.zero_grad()

		# Train
		predictions = model(X_train_tensor)
		
		print(f"Predictions shape before mean: {predictions.shape}") 

		predictions = predictions.mean(dim=1).reshape(-1, 1)

		# Print shapes before loss calculation
		print(f"Predictions shape before loss: {predictions.shape}")
		print(f"y_train_tensor shape before loss: {y_train_tensor.view(-1, 1).shape}")

		loss = loss_fn(predictions, y_train_tensor.view(-1, 1))
		loss.backward()
		optimizer.step()

		# Print loss every 10 epochs  
		if epoch % 10 == 0:
			print()
			print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

	# Evaluate
	model.eval()
	preds = model(X_test_tensor).detach().numpy()
	mse = mean_squared_error(y_test, preds)
	r2 = r2_score(y_test, preds)

	print('Test MSE:', mse)
	print('Test R-squared:', r2)