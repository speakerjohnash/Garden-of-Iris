import sys
import torch
import torch.nn as nn 
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  
from torch.optim import Adam
from datetime import datetime

from time2vec import Time2VecEncoder  
from time_transformer import TimeEncodingTransformer
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
	'Null': NullEncoder() 
}

# List of encoders you want to include
encoders = ['SinusoidalBasic', 'Null']  

# Load data
df = pd.read_csv('synthetic_data.csv')
df['Date'] = pd.to_datetime(df['Date']) 
df = df[['Date', 'Close']]
use_ordered_data = False
ensure_preceding_day = True
window_size = 5
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

	X_train_flat = X_train_encoded.flatten(start_dim=1)
	X_test_flat = X_test_encoded.flatten(start_dim=1)

	print(f"X_train_flat shape: {X_train_flat.shape}")
	print(f"X_test_flat shape: {X_test_flat.shape}")

	X_train_flat = X_train_flat.detach().numpy()
	X_test_flat = X_test_flat.detach().numpy()

	print(f"X_train_flat dtype: {X_train_flat.dtype}")
	print(f"X_test_flat dtype: {X_test_flat.dtype}")

	X_train_combined = np.hstack([X_train, X_train_flat])
	X_test_combined = np.hstack([X_test, X_test_flat])

	print(f"X_train_combined shape: {X_train_combined.shape}")
	print(f"X_test_combined shape: {X_test_combined.shape}")

	X_train_tensor = torch.tensor(X_train_combined, dtype=torch.float32)
	X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32)  
	y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
	y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

	print(f"X_train_tensor shape: {X_train_tensor.shape}")
	print(f"X_test_tensor shape: {X_test_tensor.shape}")
	print(f"y_train_tensor shape: {y_train_tensor.shape}")
	print(f"y_test_tensor shape: {y_test_tensor.shape}")

	# Sequential model for benchmark
	model = nn.Sequential(
		nn.Linear(X_train_combined.shape[1], 128),
		nn.ReLU(),
		nn.Linear(128, 64), 
		nn.ReLU(),
		nn.Linear(64, 1)
	)

	# model = TimeEncodingTransformer(input_dim=X_train_combined.shape[1])

	optimizer = Adam(model.parameters(), lr=0.001)
	loss_fn = nn.MSELoss()

	# Train  
	model.train()
	epochs = 250

	for epoch in range(epochs):
		optimizer.zero_grad()
		predictions = model(X_train_tensor)
		loss = loss_fn(predictions, y_train_tensor)
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