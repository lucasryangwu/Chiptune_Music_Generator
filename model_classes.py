from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


#------------------------------------------------

# Tonnetz Dataset

class TonnetzSequenceDataset(Dataset):
    """
    Precomputed: list of tonnetz sequences (numpy arrays of shape (T, R, C))
    We will extract sliding windows: given sequence length seq_len, we produce:
      X: (seq_len, R, C), y: (R, C) the next slice after the seq
    """
    def __init__(self, sequences: List[np.ndarray], seq_len: int = 16):
        self.samples = []
        self.seq_len = seq_len
        for seq in sequences:
            T = seq.shape[0]
            if T <= seq_len:
                continue
            # sliding windows
            for i in range(0, T - seq_len):
                X = seq[i:i+seq_len]      # (seq_len, R, C)
                y = seq[i+seq_len]        # (R, C)
                self.samples.append((X, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        # return tensors: (seq_len, channels=1, H, W) and (1, H, W)
        Xt = torch.tensor(X).unsqueeze(1)  # 1 channel
        yt = torch.tensor(y).unsqueeze(0)
        return Xt, yt

#------------------------------------------------

# Autoencoder CNN

class ConvAutoencoder(nn.Module):
    """2-layer convolutional encoder + FC decoder to reconstruct tonnetz image"""
    def __init__(self, in_channels=1, feat_maps=(20,10), rows=24, cols=12, latent_dim=128):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, feat_maps[0], kernel_size=3, padding=1)  # preserve shape
        self.enc_pool1 = nn.MaxPool2d((2,2))   # reduce
        self.enc_conv2 = nn.Conv2d(feat_maps[0], feat_maps[1], kernel_size=3, padding=1)
        # second pooling (as paper: 2x1)
        self.enc_pool2 = nn.MaxPool2d((2,1))
        # compute shape after conv/pool with given rows, cols
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, rows, cols)
            x = self.enc_conv1(dummy); x = self.enc_pool1(x)
            x = self.enc_conv2(x); x = self.enc_pool2(x)
            self.enc_out_shape = x.shape  # (1, C, H, W)
            enc_flat = int(np.prod(self.enc_out_shape[1:]))
        self.fc_enc = nn.Linear(enc_flat, latent_dim)
        # Decoder: mirror
        self.fc_dec = nn.Linear(latent_dim, enc_flat)
        self.dec_convT1 = nn.ConvTranspose2d(feat_maps[1], feat_maps[0], kernel_size=3, padding=1)
        self.unpool1 = nn.Upsample(scale_factor=(2,1), mode='nearest')
        self.dec_convT2 = nn.ConvTranspose2d(feat_maps[0], in_channels, kernel_size=3, padding=1)
        self.unpool2 = nn.Upsample(scale_factor=(2,2), mode='nearest')
        self.activation = nn.ReLU()

    def encode(self, x):
        x = self.activation(self.enc_conv1(x))
        x = self.enc_pool1(x)
        x = self.activation(self.enc_conv2(x))
        x = self.enc_pool2(x)
        batch = x.shape[0]
        flat = x.view(batch, -1)
        z = self.fc_enc(flat)
        return z

    def decode(self, z):
        batch = z.shape[0]
        x = self.fc_dec(z)
        x = x.view(batch, *tuple(self.enc_out_shape[1:]))  # (B, C, H, W)
        x = self.activation(self.dec_convT1(x))
        x = self.unpool1(x)
        x = self.activation(self.dec_convT2(x))
        x = self.unpool2(x)
        # final reconstruction logits (no sigmoid here)
        return x

    def forward(self, x):
        z = self.encode(x)
        recon_logits = self.decode(z)
        return recon_logits, z

#------------------------------------------------

class SequencePredictor(nn.Module):
    """
    LSTM sequence model that takes latent vectors sequence and predicts next frame (tonnetz)
    """
    def __init__(self, latent_dim=128, hidden_dim=256, num_layers=2, out_size=(1,24,12)):
        super().__init__()
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, np.prod(out_size))  # predict entire tonnetz image flattened
        self.out_size = out_size
    def forward(self, z_seq):
        # z_seq: (B, seq_len, latent_dim)
        output, (h_n, c_n) = self.lstm(z_seq)  # output (B, seq_len, hidden_dim)
        last = output[:, -1, :]  # take last output
        logits = self.fc(last)
        logits = logits.view(-1, *self.out_size)  # (B, 1, H, W)
        return logits

#------------------------------------------------