import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.optim import Adam

import yfinance as yf
from datetime import datetime

# Timestamp encoders
from time2vec import Time2VecEncoder
import sinusoidal_basic as basic
import sinusoidal_frequencies as freq
import iris_time_enc as iris

# Initialize encoders with appropriate parameters
t2v_encoder = Time2VecEncoder(in_features=1, out_features=6)
basic_encoder = basic.SinusoidalBasicEncoder()
freq_encoder = freq.SinusoidalFrequenciesEncoder()
iris_encoder = iris.IrisTimeEncoder(in_features=10, out_features=6)
time_encoders = [t2v_encoder, basic_encoder, freq_encoder, iris_encoder]

# Load Apple stock data
df = yf.download('AAPL', start='2017-01-01', end='2018-01-01')
df.reset_index(inplace=True)

# Preprocess 
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Close']]

# Create sliding windows
window_size = 30
X, y = [], []

for i in range(window_size, len(df)):
    X.append(df.loc[i-window_size:i-1, 'Close'].values)
    y.append(df.loc[i, 'Close'])

X, y = np.array(X), np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Indices to extract dates
train_idxs, test_idxs = train_test_split(range(window_size, len(df)), test_size=0.2, shuffle=False)

# Encode timestamps
start_date = df.loc[window_size-1, 'Date']
X_train_dates = df.loc[train_idxs, 'Date'].values
X_test_dates = df.loc[test_idxs, 'Date'].values

# Time2Vec
X_train_t2v = [t2v_encoder.encode(date, start_date) for date in X_train_dates]
X_test_t2v = [t2v_encoder.encode(date, start_date) for date in X_test_dates]

X_train_t2v = torch.stack(X_train_t2v)
X_test_t2v = torch.stack(X_test_t2v)

# Sinusoidal Basic
X_train_basic = [basic_encoder.encode(start_date, date) for date in X_train_dates]
X_test_basic = [basic_encoder.encode(start_date, date) for date in X_test_dates]

X_train_basic = torch.stack(X_train_basic)
X_test_basic = torch.stack(X_test_basic)

# Sinusoidal Frequencies
X_train_freq = [freq_encoder.encode(start_date, date) for date in X_train_dates]
X_test_freq = [freq_encoder.encode(start_date, date) for date in X_test_dates]

X_train_freq = torch.stack(X_train_freq)
X_test_freq = torch.stack(X_test_freq)

# Iris Time Encoding
X_train_iris = [iris_encoder.encode(start_date, date) for date in X_train_dates]
X_test_iris = [iris_encoder.encode(start_date, date) for date in X_test_dates]

X_train_iris = torch.stack(X_train_iris)
X_test_iris = torch.stack(X_test_iris)

# Concatenate timestamp embeddings with input data
X_trains = [X_train_t2v, X_train_basic, X_train_freq, X_train_iris] 
X_tests = [X_test_t2v, X_test_basic, X_test_freq, X_test_iris]

# Train simple MLP 
encoders = ['Time2Vec', 'SinusoidalBasic', 'SinusoidalFrequencies', 'IrisTime']

for i, (X_train_encoded, X_test_encoded) in enumerate(zip(X_trains, X_tests)):
  
    print(f"Results for {encoders[i]}:")

    size = X_train_encoded.shape[-1]

    print(f"Encoding shape: {X_train_encoded.shape}")
    print(f"Encoding size: {size}")
    print(f"X_train shape: {X_train.shape}")

    # Flatten the encoded data along all dimensions except the first (batch)
    X_train_flat = X_train_encoded.flatten(start_dim=1)
    X_test_flat = X_test_encoded.flatten(start_dim=1)

    print(f"X_train_flat shape: {X_train_flat.shape}")
    print(f"X_test_flat shape: {X_test_flat.shape}")

    # Convert to numpy arrays
    X_train_flat = X_train_flat.detach().numpy()
    X_test_flat = X_test_flat.detach().numpy()

    print(f"X_train_flat dtype: {X_train_flat.dtype}")
    print(f"X_test_flat dtype: {X_test_flat.dtype}")

    # Concatenate encoded data with original data
    X_train_combined = np.hstack([X_train, X_train_flat])
    X_test_combined = np.hstack([X_test, X_test_flat])

    print(f"X_train_combined shape: {X_train_combined.shape}")
    print(f"X_test_combined shape: {X_test_combined.shape}")

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_combined, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    print(f"X_train_tensor shape: {X_train_tensor.shape}")
    print(f"X_test_tensor shape: {X_test_tensor.shape}")
    print(f"y_train_tensor shape: {y_train_tensor.shape}")
    print(f"y_test_tensor shape: {y_test_tensor.shape}")

    model = nn.Sequential(
        nn.Linear(X_train_combined.shape[1], 128), 
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    print(f"Model input size: {X.shape[1] + size}")

    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    
    # Train 
    model.train()
    for epoch in range(1000): # Training for 100 epochs
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = loss_fn(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    preds = model(X_test_tensor).detach().numpy()
    mse = mean_squared_error(y_test_tensor, preds)
    
    print('Test MSE:', mse)