import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.optim import Adam
from datetime import datetime

# Import your timestamp encoders
from time2vec import Time2VecEncoder
from time_transformer import TimeEncodingTransformer
import sinusoidal_basic as basic
import sinusoidal_frequencies as freq
import mixed_time_enc as iris

# Null Encoder
class NullEncoder:
    def encode(self, start_date, date):
        return torch.tensor([])

# Initialize encoders with appropriate parameters
t2v_encoder = Time2VecEncoder(in_features=1, out_features=6)
basic_encoder = basic.SinusoidalBasicEncoder()
freq_encoder = freq.SinusoidalFrequenciesEncoder()
iris_encoder = iris.IrisTimeEncoder(in_features=10, out_features=6)
null_encoder = NullEncoder()
time_encoders = [t2v_encoder, basic_encoder, freq_encoder, iris_encoder, null_encoder]

# Load data
# df = yf.download('MSFT', start='2010-01-01', end='2023-01-01')
# df.reset_index(inplace=True)
df = pd.read_csv('synthetic_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Close']]

# Option to use ordered data (True) or random selection (False)
use_ordered_data = True
ensure_preceding_day = True

# Create sliding windows with a 30-day future target
window_size = 50
gap = 1  # The gap between the context window and the target
X, y = [], []

for target_idx in range(window_size, len(df) - gap):

    # Create a sliding window of the 50 days preceding the target date
    if use_ordered_data:
        preceding_idxs = range(target_idx - window_size, target_idx)
    else:
        # Randomly sample 49 dates from all available preceding dates (excluding the day right before the target)
        preceding_idxs = np.random.choice(range(target_idx - 1), window_size - 1, replace=False)

        # If ensure_preceding_day flag is set, include the day right before the target date
        if ensure_preceding_day:
            preceding_idxs = np.append(preceding_idxs, target_idx - 1)
            np.random.shuffle(preceding_idxs)  # Shuffle to randomize the position of the preceding day
    
    preceding_closes = df.loc[preceding_idxs, 'Close'].values
    target_close = df.loc[target_idx + gap, 'Close']

    X.append(preceding_closes)
    y.append(target_close)

X, y = np.array(X), np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train-test split for indices
train_idxs, test_idxs = train_test_split(range(window_size, len(df) - gap), test_size=0.2, shuffle=False)

# Encode timestamps
start_date = df.loc[window_size-1, 'Date']
X_train_dates = [df.loc[idx + gap, 'Date'] for idx in train_idxs]
X_test_dates = [df.loc[idx + gap, 'Date'] for idx in test_idxs]

# Encoding using different techniques
X_train_t2v = torch.stack([t2v_encoder.encode(date, start_date) for date in X_train_dates])
X_test_t2v = torch.stack([t2v_encoder.encode(date, start_date) for date in X_test_dates])
X_train_basic = torch.stack([basic_encoder.encode(start_date, date) for date in X_train_dates])
X_test_basic = torch.stack([basic_encoder.encode(start_date, date) for date in X_test_dates])
X_train_freq = torch.stack([freq_encoder.encode(start_date, date) for date in X_train_dates])
X_test_freq = torch.stack([freq_encoder.encode(start_date, date) for date in X_test_dates])
X_train_iris = torch.stack([iris_encoder.encode(start_date, date) for date in X_train_dates])
X_test_iris = torch.stack([iris_encoder.encode(start_date, date) for date in X_test_dates])
X_train_null = torch.stack([null_encoder.encode(start_date, date) for date in X_train_dates])
X_test_null = torch.stack([null_encoder.encode(start_date, date) for date in X_test_dates])

# Dictionary mapping encoders to their corresponding datasets
encoder_data = {
    'Time2Vec': (X_train_t2v, X_test_t2v),
    'SinusoidalBasic': (X_train_basic, X_test_basic),
    'SinusoidalFrequencies': (X_train_freq, X_test_freq),
    'MixedTime': (X_train_iris, X_test_iris),
    'Null': (X_train_null, X_test_null)
}

# List of encoders you want to include
encoders = ['Time2Vec', 'SinusoidalBasic', 'SinusoidalFrequencies', 'MixedTime', 'Null']
encoders = ['Time2Vec', 'SinusoidalBasic', 'Null']

# Initialize empty lists for training and testing datasets
X_trains = []
X_tests = []

# Append corresponding datasets for selected encoders
for encoder in encoders:
    train, test = encoder_data[encoder]
    X_trains.append(train)
    X_tests.append(test)

for i, (X_train_encoded, X_test_encoded) in enumerate(zip(X_trains, X_tests)):
  
    print()
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

    # model = TimeEncodingTransformer(input_dim=X_train_combined.shape[1])

    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Train
    model.train()
    epochs = 1000

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = loss_fn(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Print the loss every 10 epochs
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
