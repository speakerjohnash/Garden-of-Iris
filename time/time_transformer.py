import torch.nn as nn

class TimeEncodingTransformer(nn.Module):
    def __init__(self, input_dim, nhead=1, num_layers=1, dim_feedforward=64):
        super(TimeEncodingTransformer, self).__init__()

        print("Input Dimension: " + str(input_dim))
        print("Number of Heads: " + str(nhead))

        # Transformer layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )

        # Sequential feedforward classifier
        self.sequential_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Reshape the input to add the sequence length dimension
        x = x.unsqueeze(1) # Shape becomes (batch size, 1, d_model)

        # Transpose batch and sequence length dimensions
        x = x.transpose(0, 1) # Shape becomes (1, batch size, d_model)

        # Pass through the transformer
        x_encoded = self.transformer_encoder(x)

        # Squeeze the sequence length dimension
        x_encoded = x_encoded.squeeze(0)

        # Pass through the sequential feedforward classifier
        output = self.sequential_classifier(x_encoded)

        return output


