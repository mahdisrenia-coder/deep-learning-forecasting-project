# TIMESNET-FIXED MODEL

# This is a simplified version of the original TimesNet model.
# Instead of using FFT to automatically find periods in the data,
# we manually define the periods based on the data frequency.

# We wanted to test whether FFT period discovery actually helps the model.
# By comparing this simplified version with the original TimesNet,
# we can see:
#   1. Is adaptive period discovery worth the complexity?
#   2. What is lost when we use fixed periods instead?
#   3. Can the model still perform well with simple, known periods?

# The main difference is that we removed the FFT step and replaced
# it with manually defined periods (e.g., 24 for daily cycles).

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
# We do not import torch.fft here! no need for FFT from torch



# FIXED TIMES BLOCK
# the core building block of TimesNet-Fixed.
# instead of fft we use pre-defined periods in the class, no need of fft def anymore
class FixedTimesBlock(nn.Module):
    
    def __init__(self, configs):
        super(FixedTimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # ----- REPLACED FFT WITH FIXED PERIODS -----
        # Original TimesNet used FFT to find periods dynamically (configs.top_k).
        # We simplified it: we manually define periods based on data frequency.
        
        # For hourly data (ETTh1, ETTh2): 24 points = 1 day → periods that divide 24
        # For 15-min data (ECL): 96 points = 1 day → periods that divide 96
        
        # ATTENTION: ECL uses --freq 'h' but is actually 15-minute data,
        # so we check the dataset name as well.

        if 'electricity' in configs.data_path or configs.freq == 't':
            self.fixed_periods = [96, 48, 32, 24, 16]   # 15-minute data
        else:
            self.fixed_periods = [24, 12, 8, 6, 4]      # Hourly data
        
        # ----- SAME AS ORIGINAL TIMESNET (unchanged) -----
        # Shared inception blocks for 2D convolutions.
        # This part is exactly the same as the original TimesNet.
        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model,      # Input channels
                configs.d_ff,         # Hidden channels
                num_kernels=configs.num_kernels  # Number of different kernel sizes
            ),
            nn.GELU(),                # Activation function
            Inception_Block_V1(
                configs.d_ff,         # Input channels
                configs.d_model,      # Output channels (back to original size)
                num_kernels=configs.num_kernels
            )
        )

def forward(self, x):
    """
    Forward pass for TimesNet-Fixed.
    
    This is the same as original TimesNet, with only two simplifications:
        1. Periods are FIXED (no FFT discovery) → we use self.fixed_periods
        2. Period weights are UNIFORM (no FFT amplitudes) → all periods equally important
    
    Everything else (reshape, 2D convolution, residual connection) is unchanged.
    """
    B, T, N = x.size()
    res = []
    
    # ----- Process each fixed period -----
    # Original TimesNet used FFT to find periods dynamically.
    # We replaced that with manually defined periods (e.g., [24, 12, 8, 6, 4]).
    # The rest of the logic is exactly the same.
    for period in self.fixed_periods:
        if period > T:
            continue
        
        # Pad to make sequence divisible by period
        if (self.seq_len + self.pred_len) % period != 0:
            length = (((self.seq_len + self.pred_len) // period) + 1) * period
            padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N]).to(x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = (self.seq_len + self.pred_len)
            out = x
        
        # Reshape 1D → 2D: [B, rows, cols, N] → [B, N, rows, cols]
        # rows = number of cycles, cols = period length
        out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
        
        # 2D convolution (same as original)
        out = self.conv(out)
        
        # Reshape back to 1D
        out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
        res.append(out[:, :(self.seq_len + self.pred_len), :])
    
    # ----- Aggregate results from all periods -----
    # Original TimesNet used FFT amplitudes as weights (adaptive).
    # We simplified this: all periods get equal weight (uniform).
    # This tests whether FFT-based weighting is actually necessary.
    res = torch.stack(res, dim=-1)
    period_weight = torch.ones(B, self.k).to(x.device) / self.k
    period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
    res = torch.sum(res * period_weight, -1)
    
    # Residual connection (same as original)
    return res + x



# This is the same as the original TimesNet Model class,
# but it uses FixedTimesBlock instead of TimesBlock.

# The architecture is:
#   1. Embedding (positional + temporal)
#   2. Sequence expansion (linear layer)
#   3. Multiple FixedTimesBlocks (with LayerNorm)
#   4. Output projection (linear layer)


class Model(nn.Module):
# This is the same as the original TimesNet Model class,
# but it uses FixedTimesBlock instead of TimesBlock.

# The architecture is:
#   1. Embedding (positional + temporal)
#   2. Sequence expansion (linear layer)
#   3. Multiple FixedTimesBlocks (with LayerNorm)
#   4. Output projection (linear layer)

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        ## stack multiple times blocks
        self.model = nn.ModuleList([FixedTimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        
        ## Embedding layer: converts raw data to d_model dimensions
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Expands seq_len → seq_len + pred_len for multi-period analysis
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        
        #Maps d_model → c_out (number of output features)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
