import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset
import datetime
import matplotlib.pyplot as plt
import math

# Device selection
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Timestamp encoding
class TimestampEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(1, d_model // 2)
        self.frequencies = nn.Parameter(torch.rand(d_model // 4) * 10)
        self.phase_shifts = nn.Parameter(torch.rand(d_model // 4) * 2 * np.pi)

    def forward(self, x):
        x = x.unsqueeze(-1).float()
        linear_part = self.linear(x)
        sin_part = torch.sin(x * self.frequencies + self.phase_shifts)
        cos_part = torch.cos(x * self.frequencies + self.phase_shifts)
        return torch.cat([linear_part, sin_part, cos_part], dim=-1)

# Synthetic data generation
def generate_synthetic_data(num_days):
    start_date = np.datetime64('2020-01-01')
    dates = np.array([start_date + np.timedelta64(i, 'D') for i in range(num_days)])
    
    time_feature = np.arange(num_days) / 365.0
    trend = 0.5 * time_feature
    seasonal = np.sin(2 * np.pi * time_feature) + 0.5 * np.sin(4 * np.pi * time_feature)
    noise = 0.1 * np.random.randn(num_days)
    values = trend + seasonal + noise
    
    return dates, values

# Sampling function for ordered data
def sample_ordered(dates, values, window_size, target_offset):
    num_samples = len(dates) - window_size - target_offset
    X = np.zeros((num_samples, window_size))
    X_dates = np.zeros((num_samples, window_size), dtype='datetime64[D]')
    y = np.zeros(num_samples)
    y_dates = np.zeros(num_samples, dtype='datetime64[D]')
    
    for i in range(num_samples):
        X[i] = values[i:i+window_size]
        X_dates[i] = dates[i:i+window_size]
        y[i] = values[i+window_size+target_offset-1]
        y_dates[i] = dates[i+window_size+target_offset-1]
    
    return X, X_dates, y, y_dates

# Sampling function for random context
def sample_random(dates, values, context_size, num_samples):
    X = np.zeros((num_samples, context_size))
    X_dates = np.zeros((num_samples, context_size), dtype='datetime64[D]')
    y = np.zeros(num_samples)
    y_dates = np.zeros(num_samples, dtype='datetime64[D]')
    
    for i in range(num_samples):
        target_idx = np.random.randint(context_size, len(dates)-1)
        y[i] = values[target_idx]
        y_dates[i] = dates[target_idx]
        
        # Select one point close to the target
        close_idx = target_idx - 1
        
        # Select remaining points randomly from earlier in the sequence
        other_indices = np.random.choice(range(0, close_idx), context_size-1, replace=False)
        
        all_indices = np.append(other_indices, close_idx)
        
        # Shuffle the indices to randomize the order
        np.random.shuffle(all_indices)
        
        X[i] = values[all_indices]
        X_dates[i] = dates[all_indices]
    
    return X, X_dates, y, y_dates

# Visualization function
def visualize_data(dates, values, ordered_X, ordered_X_dates, ordered_y, ordered_y_dates, 
                   random_X, random_X_dates, random_y, random_y_dates):
    plt.figure(figsize=(20, 10))
    plt.plot(dates, values)
    plt.title('Full Synthetic Dataset')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()
    
    # Plot ordered samples
    plt.figure(figsize=(20, 15))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(ordered_X_dates[i], ordered_X[i], 'b-o', label='Input')
        plt.plot(ordered_y_dates[i], ordered_y[i], 'ro', label='Target')
        plt.title(f'Ordered Sample {i+1}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot random samples (sorted for visualization)
    plt.figure(figsize=(20, 15))
    for i in range(3):
        sort_indices = np.argsort(random_X_dates[i])
        plt.subplot(3, 1, i+1)
        plt.plot(random_X_dates[i][sort_indices], random_X[i][sort_indices], 'b-o', label='Input')
        plt.plot(random_y_dates[i], random_y[i], 'ro', label='Target')
        plt.title(f'Random Sample {i+1} (Sorted for Visualization)')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
    plt.tight_layout()
    plt.show()

# Specialized Attention
class TemporalAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Attention calculation
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return output

class TemporalTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = TemporalAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, use_timestamp_encoding=True):
        super().__init__()
        self.use_timestamp_encoding = use_timestamp_encoding
        self.value_embedding = nn.Linear(1, d_model)
        if use_timestamp_encoding:
            self.timestamp_encoding = TimestampEncoding(d_model)
            self.d_model = d_model * 2
        else:
            self.d_model = d_model
        self.temporal_transformer_layers = nn.ModuleList([TemporalTransformerLayer(self.d_model, nhead, self.d_model * 4) for _ in range(num_layers)])
        self.fc = nn.Linear(self.d_model, 1)
    
    def forward(self, x, timestamps, target_timestamp):
        x = self.value_embedding(x.unsqueeze(-1))
        if self.use_timestamp_encoding:
            t = self.timestamp_encoding(timestamps)
            x = torch.cat([x, t], dim=-1)  # Concatenate instead of add
        
        for layer in self.temporal_transformer_layers:
            x = layer(x)
        
        return self.fc(x[:, -1, :]).squeeze(-1)

# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, timestamps, targets, target_timestamps):
        self.data = torch.FloatTensor(data)
        self.timestamps = torch.FloatTensor(timestamps.astype(int))
        self.targets = torch.FloatTensor(targets)
        self.target_timestamps = torch.FloatTensor(target_timestamps.astype(int))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.timestamps[idx], self.targets[idx], self.target_timestamps[idx]

# Training function
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        data, timestamps, targets, target_timestamps = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(data, timestamps, target_timestamps)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            data, timestamps, targets, target_timestamps = [b.to(device) for b in batch]
            outputs = model(data, timestamps, target_timestamps)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Main experiment
def run_experiment():
    # Parameters
    num_days = 1000
    window_size = 50
    target_offset = 1
    context_size = 10
    num_samples = 10000
    d_model = 64
    nhead = 4
    num_layers = 2
    batch_size = 64
    num_epochs = 500

    # Generate data
    dates, values = generate_synthetic_data(num_days)

    # Sample ordered and random data
    ordered_X, ordered_X_dates, ordered_y, ordered_y_dates = sample_ordered(dates, values, window_size, target_offset)
    random_X, random_X_dates, random_y, random_y_dates = sample_random(dates, values, context_size, num_samples)

    # Create datasets
    ordered_dataset = TimeSeriesDataset(ordered_X, ordered_X_dates, ordered_y, ordered_y_dates)
    random_dataset = TimeSeriesDataset(random_X, random_X_dates, random_y, random_y_dates)

    # Verify random ordering
    random_sample = random_dataset[0]
    print("Random sample timestamps:")
    print(random_sample[1])

    # Split datasets
    random_train_size = int(0.8 * len(random_dataset))
    random_test_size = len(random_dataset) - random_train_size
    random_train, random_test = torch.utils.data.random_split(random_dataset, [random_train_size, random_test_size])

    ordered_train_size = int(0.8 * len(ordered_dataset))
    ordered_test_size = len(ordered_dataset) - ordered_train_size
    ordered_train, ordered_test = torch.utils.data.random_split(ordered_dataset, [ordered_train_size, ordered_test_size])

    # Create data loaders
    random_train_loader = DataLoader(random_train, batch_size=batch_size, shuffle=True)
    random_test_loader = DataLoader(random_test, batch_size=batch_size)
    ordered_train_loader = DataLoader(ordered_train, batch_size=batch_size, shuffle=True)
    ordered_test_loader = DataLoader(ordered_test, batch_size=batch_size)

    # Create models
    model_with_encoding = TimeSeriesTransformer(d_model, nhead, num_layers, use_timestamp_encoding=True).to(device)
    model_without_encoding = TimeSeriesTransformer(d_model, nhead, num_layers, use_timestamp_encoding=False).to(device)

    # Optimizers and loss function
    optimizer_with = torch.optim.Adam(model_with_encoding.parameters())
    optimizer_without = torch.optim.Adam(model_without_encoding.parameters())
    criterion = nn.MSELoss()

    # Training and evaluation
    for epoch in range(num_epochs):
        # Train and evaluate on random data
        train_loss_with_random = train_model(model_with_encoding, random_train_loader, optimizer_with, criterion, device)
        train_loss_without_random = train_model(model_without_encoding, random_train_loader, optimizer_without, criterion, device)
        
        test_loss_with_random = evaluate_model(model_with_encoding, random_test_loader, criterion, device)
        test_loss_without_random = evaluate_model(model_without_encoding, random_test_loader, criterion, device)
        
        # Train and evaluate on ordered data
        train_loss_with_ordered = train_model(model_with_encoding, ordered_train_loader, optimizer_with, criterion, device)
        train_loss_without_ordered = train_model(model_without_encoding, ordered_train_loader, optimizer_without, criterion, device)
        
        test_loss_with_ordered = evaluate_model(model_with_encoding, ordered_test_loader, criterion, device)
        test_loss_without_ordered = evaluate_model(model_without_encoding, ordered_test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("Random Data:")
        print(f"With encoding - Train Loss: {train_loss_with_random:.4f}, Test Loss: {test_loss_with_random:.4f}")
        print(f"Without encoding - Train Loss: {train_loss_without_random:.4f}, Test Loss: {test_loss_without_random:.4f}")
        print("Ordered Data:")
        print(f"With encoding - Train Loss: {train_loss_with_ordered:.4f}, Test Loss: {test_loss_with_ordered:.4f}")
        print(f"Without encoding - Train Loss: {train_loss_without_ordered:.4f}, Test Loss: {test_loss_without_ordered:.4f}")
        print()

if __name__ == "__main__":
    run_experiment()