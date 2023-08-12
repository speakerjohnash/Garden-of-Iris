import torch
import torch.nn as nn

class TimeEncodingTransformer(nn.Module):
    def __init__(self, input_dim):
        super(TimeEncodingTransformer, self).__init__()
        
        self.embedding_dim = 128
        self.embedding = nn.Linear(input_dim, self.embedding_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4), 
            num_layers=3
        )

        self.fc1 = nn.Linear(self.embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        
        # print("Input shape:", x.shape) # Debug

        # Embedding
        x = x.view(-1, x.shape[-1]) # Flatten the first two dimensions
        x = self.embedding(x)
        x = x.view(x.shape[0], -1, self.embedding_dim) # Reshape back to (batch_size, seq_len, embedding_dim)
        
        # print("After embedding shape:", x.shape) # Debug

        # Transformer
        x = self.transformer_encoder(x)
        
        # print("After transformer shape:", x.shape) # Debug

        # Flattening the sequence dimension
        x = x.mean(dim=1)
        
        # print("After mean shape:", x.shape) # Debug

        # Feed-forward layers
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)

        return x



