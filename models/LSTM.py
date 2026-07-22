# ============================================================
# LSTM MODEL
# ============================================================
# LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN)
# designed to capture long-term dependencies in sequential data.
#
# Key LSTM features:
#   1. Cell state (long-term memory) + Hidden state (short-term memory)
#   2. Three gates: Input, Forget, Output
#   3. Gated mechanisms control what to remember and forget
#
# Architecture:
#   1. LSTM layer(s) to capture temporal dependencies
#   2. Fully connected layer to project to prediction length
#
# Course connection: Recurrent Neural Networks (RNN/LSTM slides)
# - Sequential data processing
# - Gated recurrent units
# - Long-term dependencies
# ============================================================

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/1409.3215 (Original LSTM paper)
    """

    def __init__(self, configs):
        """
        Initialize the LSTM model.
        
        Args:
            configs: Configuration object containing:
                - seq_len: Input sequence length
                - pred_len: Prediction horizon
                - enc_in: Number of input features
                - d_model: LSTM hidden size
                - e_layers: Number of LSTM layers
                - dropout: Dropout rate
        """
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # --- LSTM Layer(s) ---
        # input_size: Number of features per time step (enc_in)
        # hidden_size: Dimension of the hidden state (d_model)
        # num_layers: Stacked LSTMs (e_layers)
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout
        )

        # --- Output Projection ---
        # Takes the last hidden state and projects to prediction length
        self.fc = nn.Linear(configs.d_model, self.pred_len * self.enc_in)

    def encoder(self, x):
        """
        Core encoding logic: LSTM → Linear projection.
        
        Args:
            x: Input sequence [B, seq_len, enc_in]
            
        Returns:
            pred: Predictions [B, pred_len, enc_in]
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Extract the last hidden state
        # hidden: [num_layers, B, hidden_size]
        # hidden[-1] takes the last layer's hidden state: [B, hidden_size]
        last_hidden = hidden[-1]

        # Project to prediction length
        pred = self.fc(last_hidden)  # [B, pred_len * enc_in]

        # Reshape to [B, pred_len, enc_in]
        pred = pred.view(-1, self.pred_len, self.enc_in)

        return pred

    def forecast(self, x_enc):
        """Forecasting forward pass."""
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass for long-term forecasting.
        
        Args:
            x_enc: Input sequence [B, seq_len, enc_in]
            x_mark_enc: Time features (unused in LSTM)
            x_dec: Decoder input (unused)
            x_mark_dec: Decoder time features (unused)
            mask: Masking (unused)
            
        Returns:
            dec_out: Predictions [B, pred_len, enc_in]
        """
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, enc_in]