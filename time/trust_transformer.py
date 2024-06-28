import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset
import datetime
import matplotlib.pyplot as plt
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
    
    def visualize_predictions(self, feature_index=0, split_time=None):
        predictions = self.generate_predictions()
        plt.figure(figsize=(20, 10))
        plt.plot(self.true_data[0, :, feature_index], label='True Data', color='black', linewidth=2)
        
        for period in range(self.num_time_periods):
            start, end = self.time_period_boundaries[period], self.time_period_boundaries[period+1]
            expert_source = self.expert_sources[period]
            expert_color = self.colors[expert_source % len(self.colors)]
            
            # Plot expert source
            plt.plot(range(start, end), predictions[expert_source, 0, start:end, feature_index], 
                     color=expert_color, linewidth=2, label=f'Expert (Source {expert_source+1})' if period == 0 else "")
            
            # Plot non-expert sources
            for source in range(self.num_sources):
                if source != expert_source:
                    source_color = self.colors[source % len(self.colors)]
                    lighter_color = mcolors.to_rgba(source_color, alpha=0.3)
                    plt.plot(range(start, end), predictions[source, 0, start:end, feature_index], 
                             color=lighter_color, linewidth=1, label=f'Source {source+1}' if period == 0 else "")
        
        for boundary in self.time_period_boundaries[1:-1]:
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        if split_time is not None:
            plt.axvline(x=split_time, color='red', linestyle='-', linewidth=2, label='Train/Test Split')
        
        plt.title(f'True Data vs Source Predictions (Feature {feature_index})')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

# New sampling function for source and temporal data
def sample_trust_data(true_data, source_predictions, context_size, min_future_distance, max_future_distance, num_samples):
    num_sources = source_predictions.shape[0]
    data_length = true_data.shape[1]
    
    X_context = np.zeros((num_samples, max(context_size, 1)))  # Ensure at least 1 dimension
    X_predictions = np.zeros((num_samples, 1, num_sources))  # Only one future point
    y = np.zeros(num_samples)
    
    timestamps_context = np.zeros((num_samples, max(context_size, 1)), dtype=int)
    timestamps_predictions = np.zeros((num_samples, 1), dtype=int)  # Only one future timestamp
    target_timestamps = np.zeros(num_samples, dtype=int)
    
    source_ids = np.zeros((num_samples, max(context_size, 1) + 1), dtype=int)  # +1 for the single future point
    
    for i in range(num_samples):
        if context_size > 0:
            # Randomly select context points
            context_indices = np.sort(np.random.choice(data_length - max_future_distance, context_size, replace=False))
            X_context[i, :context_size] = true_data[0, context_indices, 0]
            timestamps_context[i, :context_size] = context_indices
            source_ids[i, :context_size] = 0
        
        # Select random future point
        if context_size > 0:
            future_distance = np.random.randint(min_future_distance, max_future_distance + 1)
            future_index = context_indices[-1] + future_distance
        else:
            future_index = np.random.randint(data_length - max_future_distance, data_length)
        
        # Extract future predictions from each source
        X_predictions[i, 0, :] = source_predictions[:, 0, future_index, 0]
        
        # Set target (ground truth for the future timestamp)
        y[i] = true_data[0, future_index, 0]
        
        # Set timestamps for predictions and target
        timestamps_predictions[i, 0] = future_index
        target_timestamps[i] = future_index
        
        # Set source IDs for predictions
        source_ids[i, -1] = np.random.randint(1, num_sources + 1)
    
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
    
    def forward(self, x_context, x_predictions, timestamps_context, timestamps_predictions, source_ids):
        # Embed context
        if x_context.size(1) > 0:
            x_context = self.value_embedding(x_context.unsqueeze(-1))
            embeddings = [x_context]
            
            if self.use_temporal:
                t_context = self.timestamp_encoding(timestamps_context)
                embeddings.append(t_context)
            
            if self.use_source:
                s_context = self.source_embedding(source_ids[:, :x_context.size(1)])
                embeddings.append(s_context)
            
            x_context = torch.cat(embeddings, dim=-1)
        else:
            x_context = torch.empty(x_predictions.size(0), 0, self.d_model, device=x_predictions.device)
        
        # Embed predictions
        x_predictions = self.value_embedding(x_predictions.unsqueeze(-1))
        embeddings = [x_predictions]
        
        if self.use_temporal:
            t_predictions = self.timestamp_encoding(timestamps_predictions)
            embeddings.append(t_predictions.unsqueeze(1).expand_as(x_predictions))
        
        if self.use_source:
            s_predictions = self.source_embedding(source_ids[:, -x_predictions.size(1):])
            embeddings.append(s_predictions.unsqueeze(1).expand_as(x_predictions))
        
        x_predictions = torch.cat(embeddings, dim=-1)
        
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
    context_size = 10
    min_future_distance = 100
    max_future_distance = 300
    num_sources = 5
    num_samples = 10000
    d_model = 64
    nhead = 4
    num_layers = 2
    batch_size = 64
    num_epochs = 100
    num_time_periods = 50

    # Generate data
    dates, values = generate_synthetic_data(num_days)
    true_data = values.reshape(1, -1, 1)
    
    # Generate source predictions
    generator = SourcePredictionGenerator(true_data, num_sources, num_time_periods)
    source_predictions = generator.generate_predictions()

    # Split time
    split_time = int(0.8 * num_days)

    # Demonstrate source predictions
    generator.visualize_predictions(feature_index=0, split_time=split_time)

    # Create two sets of data: one with context and one without
    data_with_context = sample_trust_data(true_data, source_predictions, context_size, min_future_distance, max_future_distance, num_samples)
    data_without_context = sample_trust_data(true_data, source_predictions, 0, min_future_distance, max_future_distance, num_samples)

    # Visualize data (only for with context scenario)
    X_context, X_predictions, y, timestamps_context, timestamps_predictions, target_timestamps, source_ids = data_with_context
    visualize_data(dates, values, X_context, X_predictions, y, timestamps_context, timestamps_predictions)

    # Create datasets
    dataset_with_context = TrustDataset(*data_with_context)
    dataset_without_context = TrustDataset(*data_without_context)

    # Split datasets based on time
    def split_dataset(dataset):
        train_indices = [i for i in range(len(dataset)) if dataset.target_timestamps[i] < split_time]
        test_indices = [i for i in range(len(dataset)) if dataset.target_timestamps[i] >= split_time]
        return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, test_indices)

    train_dataset_with_context, test_dataset_with_context = split_dataset(dataset_with_context)
    train_dataset_without_context, test_dataset_without_context = split_dataset(dataset_without_context)

    # Create data loaders
    train_loader_with_context = DataLoader(train_dataset_with_context, batch_size=batch_size, shuffle=True)
    test_loader_with_context = DataLoader(test_dataset_with_context, batch_size=batch_size)
    train_loader_without_context = DataLoader(train_dataset_without_context, batch_size=batch_size, shuffle=True)
    test_loader_without_context = DataLoader(test_dataset_without_context, batch_size=batch_size)

    # Create models
    models = {
        "Baseline (Value Only)": TrustTransformer(d_model, nhead, num_layers, num_sources + 1, use_temporal=False, use_source=False).to(device),
        "Temporal Embeddings Only": TrustTransformer(d_model, nhead, num_layers, num_sources + 1, use_temporal=True, use_source=False).to(device),
        "Source Embeddings Only": TrustTransformer(d_model, nhead, num_layers, num_sources + 1, use_temporal=False, use_source=True).to(device),
        "Full Ŧrust Model": TrustTransformer(d_model, nhead, num_layers, num_sources + 1, use_temporal=True, use_source=True).to(device)
    }

    # Create optimizers and loss functions
    optimizers = {name: torch.optim.Adam(model.parameters()) for name, model in models.items()}
    criterion = nn.MSELoss()

    # Training and evaluation
    for context in ["Without Context", "With Context"]:
        print(f"\n--- Experiment {context} ---")
        train_loader = train_loader_with_context if context == "With Context" else train_loader_without_context
        test_loader = test_loader_with_context if context == "With Context" else test_loader_without_context

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            for name, model in models.items():
                train_loss = train_model(model, train_loader, optimizers[name], criterion, device)
                test_loss = evaluate_model(model, test_loader, criterion, device)
                
                print(f"{name:<25} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            print()

        # Final evaluation
        print("Final Results:")
        for name, model in models.items():
            test_loss = evaluate_model(model, test_loader, criterion, device)
            print(f"{name:<25} - Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    run_experiment()