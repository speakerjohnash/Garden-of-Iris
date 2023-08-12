import torch
import torch.nn as nn

class TimeEncodingTransformer(nn.Module):
    def __init__(self, input_dim):
        super(TimeEncodingTransformer, self).__init__()
        
        self.embedding_dim = 64 # Reduced from 128
        self.embedding = nn.Linear(input_dim, self.embedding_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4), 
            num_layers=2 # Reduced from 3
        )

        self.fc1 = nn.Linear(self.embedding_dim, 64) # Reduced from 128
        self.fc2 = nn.Linear(64, 32) # Reduced from 64
        self.fc3 = nn.Linear(32, 1) # Output layer remains the same

    def forward(self, x):
        
        # Embedding
        x = x.view(-1, x.shape[-1]) # Flatten the first two dimensions
        x = self.embedding(x)
        x = x.view(x.shape[0], -1, self.embedding_dim) # Reshape back to (batch_size, seq_len, embedding_dim)

        # Transformer
        x = self.transformer_encoder(x)

        # Flattening the sequence dimension
        x = x.mean(dim=1)

        # Feed-forward layers
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)

        return x

