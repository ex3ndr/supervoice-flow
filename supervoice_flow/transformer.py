import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
from torch.cuda.amp import autocast
from .tensors import RMSNorm, AdaptiveRMSNorm
from flash_attn import flash_attn_func

class Transformer(nn.Module):
    def __init__(self, 
        n_heads,
        n_layers,
        n_dim,
        n_dim_head,
        n_dim_ffn,
        dropout, 
        enable_skip_connections = True
    ):
        super(Transformer, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.enable_skip_connections = enable_skip_connections

        # Attention blocks
        self.layers = torch.nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(AttentionBlock(
                n_heads = n_heads, 
                n_dim = n_dim, 
                n_dim_head = n_dim_head, 
                n_dim_ffn = n_dim_ffn,
                dropout = dropout,
            ))
        
        # Skip connections
        self.skip_combiners = torch.nn.ModuleList([])
        if enable_skip_connections:
            for i in range(n_layers//2):
                self.skip_combiners.append(torch.nn.Linear(n_dim * 2, n_dim))

        # Output normalization
        self.output_norm = RMSNorm(n_dim)

        # Positional embedding
        self.register_buffer('alibi_slopes', get_slopes_power_of_2(n_heads))


    def forward(self, x, t):
        batch, seq_len, *_ = x.shape

        # Run through attention blocks
        connections = []
        for i in range(self.n_layers):

            # Skip connection
            if self.n_layers - (self.n_layers // 2) <= i and self.enable_skip_connections:
                s = connections.pop()
                x = torch.cat([x, s], dim = -1)
                x = self.skip_combiners[i - (self.n_layers // 2)](x)

            # Attention
            x = self.layers[i](x, t, alibi = self.alibi_slopes)

            # Skip connection
            if i <= self.n_layers // 2:
                connections.append(x)

        # Output normalization
        x = self.output_norm(x)

        # Result
        return x


class AttentionBlock(torch.nn.Module):
    def __init__(self, n_heads, n_dim, n_dim_head, n_dim_ffn, dropout):
        super(AttentionBlock, self).__init__()

        self.n_heads = n_heads
        self.n_dim_head = n_dim_head
        self.dropout = dropout

        # Attention input layer norm
        self.attention_ln = AdaptiveRMSNorm(n_dim)

        # Input -> Query/Key/Value for each head in single tensor for speedup
        self.attention = nn.Linear(n_dim, 3 * n_dim_head * n_heads, bias=False)
        torch.nn.init.normal_(self.attention.weight, mean=0.0, std=0.02)

        # Output flatten multiple heads into single tensor
        self.attention_output = nn.Linear(n_dim_head * n_heads, n_dim, bias=False)
        torch.nn.init.normal_(self.attention_output.weight, mean=0.0, std=0.02)

        # MLP part
        self.mlp_ln = AdaptiveRMSNorm(n_dim)
        self.mlp_input = nn.Linear(n_dim, n_dim_ffn)
        self.mlp_output = nn.Linear(n_dim_ffn, n_dim)
        self.mlp_output_dropout = nn.Dropout(dropout)

    def forward(self, x, t, alibi = None):

        B, T, C = x.size() # batch size, sequence length, context width

        # Residual
        residual = x

        # Input normalization
        y = self.attention_ln(x, cond = t)

        # Calculation Q/K/V for each head
        q, k, v = self.attention(y).chunk(3, dim = -1)

        # Attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = self.n_heads), (q, k, v))
        y = flash_attn_func(q, k, v, alibi_slopes = alibi)
        y = rearrange(y, 'b n h d -> b n (h d)', h = self.n_heads)

        # Output
        y = self.attention_output(y)

        # Residual
        y = residual + y
        residual = y

        # MLP
        y = self.mlp_ln(y, cond = t)
        y = self.mlp_input(y)
        y = F.gelu(y)
        y = self.mlp_output_dropout(y)
        y = self.mlp_output(y)
        y = residual + y

        return y

#
# AliBi implementation
#

def get_slopes_power_of_2(n_heads):
    start = (2**(-2**-(math.log2(n_heads)-3)))
    ratio = start
    return torch.tensor([start*ratio**i for i in range(n_heads)], dtype=torch.float32)