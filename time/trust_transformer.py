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

# Source encoding
class SourceEmbedding(nn.Module):
    def __init__(self, num_sources, d_model):
        super().__init__()
        self.num_sources = num_sources
        self.d_model = d_model
        self.embedding = nn.Embedding(num_sources, d_model)
        self.linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        x = self.embedding(x)
        return self.linear(x)

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

def demonstrate_source_predictions(data, timestamps, targets):
    num_sources = 5
    num_time_periods = 10
    generator = SourcePredictionGenerator(data, num_sources, num_time_periods)
    
    # Generate and visualize predictions
    generator.visualize_predictions(feature_index=0)
    
    # Print expertise levels
    print("Expertise levels for each source in each time period:")
    print(generator.expertise_levels)

    # You might want to return the generated predictions for further use
    return generator.generate_predictions()

# Visualization function
def visualize_data(dates, values, X_context, X_predictions, y, timestamps_context, timestamps_predictions):
    plt.figure(figsize=(20, 10))
    plt.plot(dates, values)
    plt.title('Full Synthetic Dataset')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()
    
    # Plot sample data points
    num_samples = min(3, len(X_context))
    plt.figure(figsize=(20, 5 * num_samples))
    
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)
        
        # Plot context window
        context_dates = dates[timestamps_context[i]]
        plt.plot(context_dates, X_context[i], 'b-o', label='Context')
        
        # Plot source predictions
        prediction_dates = dates[timestamps_predictions[i]]
        for j in range(X_predictions.shape[2]):
            plt.plot(prediction_dates, X_predictions[i, :, j], '--', alpha=0.5, label=f'Source {j+1}')
        
        # Plot true future value
        plt.plot(dates[timestamps_predictions[i][-1]], y[i], 'ro', markersize=10, label='True Future')
        
        plt.title(f'Sample {i+1}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

class SourcePredictionGenerator:
    def __init__(self, true_data, num_sources, num_time_periods):
        self.true_data = true_data
        self.num_sources = num_sources
        self.num_time_periods = num_time_periods
        self.time_period_boundaries = np.linspace(0, true_data.shape[1], num_time_periods + 1, dtype=int)
        self.expertise_levels = np.random.rand(num_sources, num_time_periods)
        
    def generate_predictions(self):
        predictions = np.zeros((self.num_sources, *self.true_data.shape))
        for source in range(self.num_sources):
            for period in range(self.num_time_periods):
                start, end = self.time_period_boundaries[period], self.time_period_boundaries[period+1]
                noise_level = 1 - self.expertise_levels[source, period]
                noise = np.random.normal(0, noise_level, size=self.true_data[:, start:end].shape)
                predictions[source, :, start:end] = self.true_data[:, start:end] + noise
        return predictions
    
    def visualize_predictions(self, feature_index=0):
        predictions = self.generate_predictions()
        plt.figure(figsize=(15, 10))
        plt.plot(self.true_data[0, :, feature_index], label='True Data', color='black', linewidth=2)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_sources))
        for i in range(min(2, self.num_sources)):
            plt.plot(predictions[i, 0, :, feature_index], label=f'Source {i+1}', color=colors[i], alpha=0.7)
        
        for boundary in self.time_period_boundaries[1:-1]:
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        plt.title(f'True Data vs Source Predictions (Feature {feature_index})')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

# New sampling function for source and temporal data
def sample_trust_data(true_data, source_predictions, context_size, future_size, num_samples):
    num_sources = source_predictions.shape[0]
    data_length = true_data.shape[1]
    
    X_context = np.zeros((num_samples, context_size))
    X_predictions = np.zeros((num_samples, future_size, num_sources))
    y = np.zeros(num_samples)
    
    timestamps_context = np.zeros((num_samples, context_size))
    timestamps_predictions = np.zeros((num_samples, future_size))
    target_timestamps = np.zeros(num_samples)
    
    source_ids = np.zeros((num_samples, context_size + future_size), dtype=int)
    
    for i in range(num_samples):
        # Randomly select the end of the context window
        context_end = np.random.randint(context_size, data_length - future_size)
        
        # Extract context window
        X_context[i] = true_data[0, context_end - context_size:context_end, 0]
        timestamps_context[i] = np.arange(context_end - context_size, context_end)
        
        # Extract future predictions from each source
        X_predictions[i] = source_predictions[:, 0, context_end:context_end + future_size, 0].T
        
        # Set target (ground truth for the last future timestamp)
        y[i] = true_data[0, context_end + future_size - 1, 0]
        
        # Set timestamps for predictions and target
        timestamps_predictions[i] = np.arange(context_end, context_end + future_size)
        target_timestamps[i] = context_end + future_size - 1
        
        # Set source IDs (0 for true data in context, 1 to num_sources for predictions)
        source_ids[i, :context_size] = 0
        source_ids[i, context_size:] = np.random.randint(1, num_sources + 1, size=future_size)
    
    return X_context, X_predictions, y, timestamps_context, timestamps_predictions, target_timestamps, source_ids

# Updated TimeSeriesDataset
class TrustDataset(Dataset):
    def __init__(self, X_context, X_predictions, y, timestamps_context, timestamps_predictions, target_timestamps, source_ids):
        self.X_context = torch.FloatTensor(X_context)
        self.X_predictions = torch.FloatTensor(X_predictions)
        self.y = torch.FloatTensor(y)
        self.timestamps_context = torch.LongTensor(timestamps_context)
        self.timestamps_predictions = torch.LongTensor(timestamps_predictions)
        self.target_timestamps = torch.LongTensor(target_timestamps)
        self.source_ids = torch.LongTensor(source_ids)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X_context[idx], self.X_predictions[idx], self.timestamps_context[idx], 
                self.timestamps_predictions[idx], self.target_timestamps[idx], self.source_ids[idx], self.y[idx])

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

# Updated TimeSeriesTransformer
class TrustTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_sources):
        super().__init__()
        self.value_embedding = nn.Linear(1, d_model)
        self.timestamp_encoding = TimestampEncoding(d_model)
        self.source_embedding = SourceEmbedding(num_sources, d_model)
        self.d_model = d_model * 3
        self.temporal_transformer_layers = nn.ModuleList([TemporalTransformerLayer(self.d_model, nhead, self.d_model * 4) for _ in range(num_layers)])
        self.fc = nn.Linear(self.d_model, 1)
    
    def forward(self, x_context, x_predictions, timestamps_context, timestamps_predictions, source_ids):
        # Embed context
        x_context = self.value_embedding(x_context.unsqueeze(-1))
        t_context = self.timestamp_encoding(timestamps_context)
        s_context = self.source_embedding(source_ids[:, :x_context.size(1)])
        x_context = torch.cat([x_context, t_context, s_context], dim=-1)
        
        # Embed predictions
        x_predictions = self.value_embedding(x_predictions.unsqueeze(-1))
        t_predictions = self.timestamp_encoding(timestamps_predictions)
        s_predictions = self.source_embedding(source_ids[:, x_context.size(1):])
        x_predictions = torch.cat([x_predictions, t_predictions.unsqueeze(1).expand_as(x_predictions), s_predictions.unsqueeze(1).expand_as(x_predictions)], dim=-1)
        
        # Combine context and predictions
        x = torch.cat([x_context, x_predictions.view(x_predictions.size(0), -1, self.d_model)], dim=1)
        
        for layer in self.temporal_transformer_layers:
            x = layer(x)
        
        return self.fc(x[:, -1, :]).squeeze(-1)

# Updated training function
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x_context, x_predictions, timestamps_context, timestamps_predictions, target_timestamps, source_ids, targets = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(x_context, x_predictions, timestamps_context, timestamps_predictions, source_ids)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Updated evaluation function
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            x_context, x_predictions, timestamps_context, timestamps_predictions, target_timestamps, source_ids, targets = [b.to(device) for b in batch]
            outputs = model(x_context, x_predictions, timestamps_context, timestamps_predictions, source_ids)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Updated run_experiment function
def run_experiment():
    # Parameters
    num_days = 1000
    context_size = 30
    future_size = 10
    num_sources = 5
    num_samples = 10000
    d_model = 64
    nhead = 4
    num_layers = 2
    batch_size = 64
    num_epochs = 100

    # Generate data
    dates, values = generate_synthetic_data(num_days)
    true_data = values.reshape(1, -1, 1)
    
    # Generate source predictions
    generator = SourcePredictionGenerator(true_data, num_sources, num_time_periods=10)
    source_predictions = generator.generate_predictions()

    # Demonstrate source predictions
    demonstrate_source_predictions(true_data, np.arange(len(values)), np.zeros(len(values)))

    # Sample data
    X_context, X_predictions, y, timestamps_context, timestamps_predictions, target_timestamps, source_ids = sample_trust_data(
        true_data, source_predictions, context_size, future_size, num_samples
    )

    # Visualize data
    visualize_data(dates, values, X_context, X_predictions, y, timestamps_context, timestamps_predictions)

    # Create dataset
    dataset = TrustDataset(X_context, X_predictions, y, timestamps_context, timestamps_predictions, target_timestamps, source_ids)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model
    model = TrustTransformer(d_model, nhead, num_layers, num_sources + 1).to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # Training and evaluation
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print()

if __name__ == "__main__":
    run_experiment()