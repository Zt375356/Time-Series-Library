import math
import torch
import torch.nn as nn


class TSTformer(nn.Module):
    def __init__(self, args):
        super(TSTformer, self).__init__()

        self.args = args
        self.num_channel = args.enc_in
        self.d_model = args.d_model
        self.num_class = args.num_class
        self.mlp_dim = args.d_ff
        self.e_layers = args.e_layers
        self.device = args.device

        self.fc1 = nn.Linear(self.num_channel, self.d_model)

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.args.n_heads,
            dim_feedforward=self.mlp_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.e_layers)

        # Classifier head
        self.classifier = nn.Linear(self.d_model, self.num_class)

    def generate_positional_encoding(self, d_model, max_len):
        """
        Generate positional encoding as done in the original Transformer paper.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe


    def classification(self, x, x_mark_enc):
        bs = x.shape[0]
        # x = x.permute(0,2,1)
        # Project input to d_model dimension
        x = self.fc1(x)

        # Generate and add positional encoding
        poscode = self.generate_positional_encoding(self.d_model, x.size(1)).to(self.device)
        x = x + poscode[:bs, :, :]

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Pooling or global aggregation method (e.g., mean pooling)
        x = torch.mean(x, dim=1)

        # Classification head
        out = self.classifier(x)

        return out
    

    def forward(self, x, x_mark_enc):
        bs = x.shape[0]
        x = x.permute(0,2,1)
        # Project input to d_model dimension
        x = self.fc1(x)

        # Generate and add positional encoding
        poscode = self.generate_positional_encoding(self.d_model, x.size(1)).to(self.device)
        x = x + poscode[:bs, :, :]

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Pooling or global aggregation method (e.g., mean pooling)
        x = torch.mean(x, dim=1)

        # Classification head
        out = self.classifier(x)

        return out
