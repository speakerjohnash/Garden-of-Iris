import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

import matplotlib.colors as mcolors

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

def generate_advanced_synthetic_data(start_date, end_date, num_samples):
    start = np.datetime64(start_date)
    end = np.datetime64(end_date)
    
    # Generate evenly spaced timestamps
    dates = np.linspace(start.astype(int), end.astype(int), num_samples).astype('datetime64[s]')
    
    total_days = (end - start).astype('timedelta64[D]').astype(int)
    time_feature = np.linspace(0, total_days / 365.25, num_samples)
    
    trend = 0.5 * time_feature
    seasonal = np.sin(2 * np.pi * time_feature) + 0.5 * np.sin(4 * np.pi * time_feature)
    pattern = trend + seasonal
    
    noise = 0.1 * np.random.randn(num_samples)
    
    values = pattern + noise
    
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
        
        # Plot context points
        context_dates = dates[timestamps_context[i]]
        plt.scatter(context_dates, X_context[i], c='blue', label='Context', zorder=3)
        
        # Plot source predictions
        prediction_date = dates[timestamps_predictions[i][0]]
        for j in range(X_predictions.shape[2]):
            plt.scatter(prediction_date, X_predictions[i, 0, j], marker='x', s=100, label=f'Source {j+1}')
        
        # Plot true future value
        plt.scatter(dates[timestamps_predictions[i][0]], y[i], c='red', s=100, label='True Future', zorder=3)
        
        # Plot true data line
        plt.plot(dates, values, c='gray', alpha=0.5, zorder=1)
        
        plt.title(f'Sample {i+1}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_no_context_data(dates, values, X_predictions, y, timestamps_predictions, source_ids, num_samples=5):
    # Subsample to 1000 points if necessary
    num_points = len(dates)
    if num_points > 1000:
        indices = np.linspace(0, num_points - 1, 1000, dtype=int)
        dates_sampled = dates[indices]
        values_sampled = values[indices]
    else:
        dates_sampled = dates
        values_sampled = values

    plt.figure(figsize=(20, 5 * num_samples))
    
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)
        
        # Plot true data line
        plt.plot(dates_sampled, values_sampled, c='gray', alpha=0.5, zorder=1, label='True Data')
        
        # Plot source predictions
        prediction_date = dates[timestamps_predictions[i]]
        for j in range(X_predictions.shape[1]):
            plt.scatter(prediction_date, X_predictions[i, j], marker='x', s=100, label=f'Source {j+1}')
        
        # Plot true future value
        plt.scatter(prediction_date, y[i], c='red', s=100, label='True Future', zorder=3)
        
        plt.title(f'Sample {i+1} (No Context)')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Print the data for the first sample
    print("\nData for the first sample:")
    print(f"Timestamp: {dates[timestamps_predictions[0]]}")
    print(f"True value: {y[0]:.4f}")
    print("Source predictions:")
    for j in range(X_predictions.shape[1]):
        print(f"  Source {j+1}: {X_predictions[0, j]:.4f}")
    print(f"Selected source ID: {source_ids[0]}")

class SourcePredictionGenerator:
    def __init__(self, true_data, num_sources, num_time_periods):
        self.true_data = true_data
        self.num_sources = num_sources
        self.num_time_periods = num_time_periods
        self.time_period_boundaries = np.linspace(0, true_data.shape[1], num_time_periods + 1, dtype=int)
        self.expert_sources = np.tile(np.arange(num_sources), (num_time_periods + num_sources - 1) // num_sources)[:num_time_periods]
        
        # Generate a custom color palette with peaceful, rich colors
        custom_colors = [
            "#3D8A44", # Forest Green
            "#4B0082",  # Indigo
            "#1E90FF",  # Dodger Blue
            "#00CED1",  # Dark Turquoise
            "#9370DB",  # Medium Purple
            "#4682B4",  # Steel Blue
            "#20B2AA",  # Light Sea Green
            "#483D8B",  # Dark Slate Blue
            "#008B8B",  # Dark Cyan
            "#5F9EA0",  # Cadet Blue
        ]
        self.colors = [mcolors.to_rgba(color) for color in custom_colors]
        
    def generate_predictions(self):
        predictions = np.zeros((self.num_sources, *self.true_data.shape))
        for period in range(self.num_time_periods):
            start, end = self.time_period_boundaries[period], self.time_period_boundaries[period+1]
            expert_source = self.expert_sources[period]
            
            for source in range(self.num_sources):
                if source == expert_source:
                    predictions[source, :, start:end] = self.true_data[:, start:end]
                else:
                    # Generate completely random predictions for non-expert sources
                    predictions[source, :, start:end] = np.random.uniform(
                        low=np.min(self.true_data),
                        high=np.max(self.true_data),
                        size=self.true_data[:, start:end].shape
                    )
        
        return predictions
    
    def visualize_predictions(self, feature_index=0, dates=None):
        predictions = self.generate_predictions()
        
        # Subsample to 1000 points
        num_points = self.true_data.shape[1]
        if num_points > 1000:
            indices = np.linspace(0, num_points - 1, 1000, dtype=int)
            true_data_sampled = self.true_data[0, indices, feature_index]
            predictions_sampled = predictions[:, 0, indices, feature_index]
            if dates is not None:
                dates_sampled = dates[indices]
        else:
            true_data_sampled = self.true_data[0, :, feature_index]
            predictions_sampled = predictions[:, 0, :, feature_index]
            dates_sampled = dates

        plt.figure(figsize=(20, 10))
        plt.plot(true_data_sampled, label='True Data', color='black', linewidth=2)
        
        for source in range(self.num_sources):
            color = self.colors[source % len(self.colors)]
            plt.plot(predictions_sampled[source], color=color, alpha=0.5, linewidth=1, label=f'Source {source+1}')
        
        if dates is not None:
            split_time = dates[0] + 0.8 * (dates[-1] - dates[0])
            split_index = np.searchsorted(dates_sampled, split_time)
            plt.axvline(x=split_index, color='red', linestyle='-', linewidth=2, label='Train/Test Split')
        
        plt.title(f'True Data vs Source Predictions (Feature {feature_index})')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

# New sampling function for source and temporal data
def sample_trust_data(true_data, source_predictions, num_samples):
    num_sources = source_predictions.shape[0]
    data_length = true_data.shape[1]
    
    X_predictions = np.zeros((num_samples, num_sources))
    y = np.zeros(num_samples)
    timestamps_predictions = np.zeros(num_samples, dtype=int)
    source_ids = np.tile(np.arange(num_sources), (num_samples, 1))
    
    for i in range(num_samples):
        future_index = np.random.randint(0, data_length)
        X_predictions[i] = source_predictions[:, 0, future_index, 0]
        y[i] = true_data[0, future_index, 0]
        timestamps_predictions[i] = future_index
    
    return X_predictions, y, timestamps_predictions, source_ids

# Updated TimeSeriesDataset
class TrustDataset(Dataset):
    def __init__(self, X_predictions, y, timestamps_predictions, source_ids):
        self.X_predictions = torch.FloatTensor(X_predictions)
        self.y = torch.FloatTensor(y)
        self.timestamps_predictions = torch.LongTensor(timestamps_predictions)
        self.source_ids = torch.LongTensor(source_ids)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X_predictions[idx], self.timestamps_predictions[idx], self.source_ids[idx], self.y[idx])

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
    def __init__(self, d_model, nhead, num_layers, num_sources, use_temporal=True, use_source=True):
        super().__init__()
        self.value_embedding = nn.Linear(1, d_model)
        self.use_temporal = use_temporal
        self.use_source = use_source
        
        if use_temporal:
            self.timestamp_encoding = TimestampEncoding(d_model)
        if use_source:
            self.source_embedding = SourceEmbedding(num_sources, d_model)
        
        self.d_model = d_model * (1 + use_temporal + use_source)
        self.temporal_transformer_layers = nn.ModuleList([TemporalTransformerLayer(self.d_model, nhead, self.d_model * 4) for _ in range(num_layers)])
        self.fc = nn.Linear(self.d_model, 1)
    
    def forward(self, x_predictions, timestamps_predictions, source_ids):
        # Embed predictions
        x = self.value_embedding(x_predictions.unsqueeze(-1))
        
        if self.use_temporal:
            t = self.timestamp_encoding(timestamps_predictions)
            t = t.unsqueeze(1).expand(-1, x_predictions.size(1), -1)
            x = torch.cat([x, t], dim=-1)
        
        if self.use_source:
            s = self.source_embedding(source_ids)
            x = torch.cat([x, s], dim=-1)
        
        for layer in self.temporal_transformer_layers:
            x = layer(x)
        
        return self.fc(x[:, -1, :]).squeeze(-1)

# Updated training function
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x_predictions, timestamps_predictions, source_ids, targets = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(x_predictions, timestamps_predictions, source_ids)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            x_predictions, timestamps_predictions, source_ids, targets = [b.to(device) for b in batch]
            outputs = model(x_predictions, timestamps_predictions, source_ids)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def evaluate_model_with_details(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    all_expert_values = []
    with torch.no_grad():
        for batch in test_loader:
            x_predictions, timestamps_predictions, source_ids, targets = [b.to(device) for b in batch]
            outputs = model(x_predictions, timestamps_predictions, source_ids)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_expert_values.extend(x_predictions[:, 0].cpu().numpy())  # Assuming first source is expert
    return total_loss / len(test_loader), all_predictions, all_targets, all_expert_values

def visualize_predictions(predictions, targets, expert_values, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5, label='Model predictions')
    plt.scatter(targets, expert_values, alpha=0.5, label='Expert values')
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', label='Perfect prediction')
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title(f'{model_name} Predictions vs True Values')
    plt.legend()
    plt.show()

def analyze_source_predictions(source_predictions):
    num_sources = source_predictions.shape[0]
    num_timepoints = source_predictions.shape[2]
    
    for source in range(num_sources):
        mean = np.mean(source_predictions[source, 0, :, 0])
        std = np.std(source_predictions[source, 0, :, 0])
        print(f"Source {source + 1}: Mean = {mean:.4f}, Std = {std:.4f}")
    
    expert_counts = np.argmax(source_predictions, axis=0)[0, :, 0]
    print(f"\nExpert source distribution:")
    for source in range(num_sources):
        count = np.sum(expert_counts == source)
        print(f"Source {source + 1}: {count} times ({count/num_timepoints*100:.2f}%)")

# Updated run_experiment function
def run_experiment():
    # Parameters
    start_date = '2020-01-01'
    end_date = '2022-09-27'
    num_days = 1000
    num_sources = 10
    num_samples = 100000
    d_model = 64
    nhead = 4
    num_layers = 2
    batch_size = 64
    num_epochs = 200
    num_time_periods = 50

    # Generate data
    # dates, values = generate_synthetic_data(num_days)
    dates, values = generate_advanced_synthetic_data(start_date, end_date, num_samples)

    true_data = values.reshape(1, -1, 1)
    
    # Generate source predictions
    generator = SourcePredictionGenerator(true_data, num_sources, num_time_periods)
    source_predictions = generator.generate_predictions()

    # Analyze source predictions
    analyze_source_predictions(source_predictions)

    # Demonstrate source predictions
    generator.visualize_predictions(feature_index=0, dates=dates)

    # Create data without context
    data = sample_trust_data(true_data, source_predictions, num_samples)

    # Visualize data for no-context scenario
    X_predictions, y, timestamps_predictions, source_ids = data
    visualize_no_context_data(dates, values, X_predictions, y, timestamps_predictions, source_ids)

    # Create dataset
    dataset = TrustDataset(*data)

    # Split dataset based on time
    def split_dataset(dataset, dates):
        split_time = dates[0] + 0.8 * (dates[-1] - dates[0])
        train_indices = [i for i in range(len(dataset)) if dates[dataset.timestamps_predictions[i]] < split_time]
        test_indices = [i for i in range(len(dataset)) if dates[dataset.timestamps_predictions[i]] >= split_time]
        return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, test_indices)

    train_dataset, test_dataset = split_dataset(dataset, dates)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create models
    models = {
        "Baseline (Value Only)": TrustTransformer(d_model, nhead, num_layers, num_sources, use_temporal=False, use_source=False).to(device),
        "Temporal Embeddings Only": TrustTransformer(d_model, nhead, num_layers, num_sources, use_temporal=True, use_source=False).to(device),
        "Source Embeddings Only": TrustTransformer(d_model, nhead, num_layers, num_sources, use_temporal=False, use_source=True).to(device),
        "Full Ŧrust Model": TrustTransformer(d_model, nhead, num_layers, num_sources, use_temporal=True, use_source=True).to(device)
    }

    # Create optimizers and loss functions
    optimizers = {name: torch.optim.Adam(model.parameters()) for name, model in models.items()}
    criterion = nn.MSELoss()

    # Initialize loss tracking
    train_losses = {name: [] for name in models.keys()}
    test_losses = {name: [] for name in models.keys()}

    # Training and evaluation
    print("\n--- Experiment ---")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for name, model in models.items():
            train_loss = train_model(model, train_loader, optimizers[name], criterion, device)
            test_loss = evaluate_model(model, test_loader, criterion, device)
            
            train_losses[name].append(train_loss)
            test_losses[name].append(test_loss)
            
            print(f"{name:<25} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        print()

    # Visualize training and test losses
    def plot_losses(losses, title):
        plt.figure(figsize=(15, 10))
        for i, (name, loss) in enumerate(losses.items(), 1):
            plt.subplot(2, 2, i)
            plt.plot(loss)
            plt.title(f"{name} - {title}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
        plt.tight_layout()
        plt.show()

    plot_losses(train_losses, "Training Loss")
    plot_losses(test_losses, "Test Loss")

    # Final evaluation
    print("Final Results:")
    for name, model in models.items():
        test_loss, predictions, targets, expert_values = evaluate_model_with_details(model, test_loader, criterion, device)
        print(f"{name:<25} - Test Loss: {test_loss:.4f}")
        print(f"  Mean Absolute Error: {np.mean(np.abs(np.array(predictions) - np.array(targets))):.4f}")
        print(f"  Mean Absolute Error (Expert): {np.mean(np.abs(np.array(expert_values) - np.array(targets))):.4f}")
        print(f"  Correlation with targets: {np.corrcoef(predictions, targets)[0, 1]:.4f}")
        print(f"  Correlation with expert: {np.corrcoef(predictions, expert_values)[0, 1]:.4f}")
        
        # Visualize predictions
        visualize_predictions(predictions, targets, expert_values, name)

if __name__ == "__main__":
    run_experiment()