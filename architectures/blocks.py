import torch 
import torch.nn as nn 
import architectures.ops as ops
from data.tensor_tools import causal_mask


class TransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, dropout_rate, submodule: ops.SwiGLU, act = nn.GELU()):

        super().__init__()

        self.attention = ops.Attention(dim, n_heads, act)
        self.submodule = submodule(dim) if submodule is not None else ops.MLP(dim, dim, (4 * dim,), act)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ada_norm1 = ops.AdaLN(dim)
        self.ada_norm2 = ops.AdaLN(dim)

        self.drop_att = nn.Dropout(dropout_rate)
        self.drop_mlp = nn.Dropout(dropout_rate)

    def forward(self, x, z=None, c=None, shift=None, mask=None):

        x = self.ada_norm1(x, z) if z is not None else self.norm1(x) 
        x = x + self.drop_att(self.attention(x, shift=shift, mask=mask))

        x = self.ada_norm2(x, c) if c is not None else self.norm2(x) 
        x = x + self.drop_mlp(self.submodule(x))

        return x


class TransformerDecoderBlock(TransformerBlock):

    def __init__(self, dim, n_heads, dropout_rate, submodule: ops.SwiGLU, act = nn.GELU()):

        super().__init__(dim, n_heads, dropout_rate, submodule, act)

        self.cross_attention = ops.CrossAttention(dim, n_heads, act)
        self.ada_norm1 = ops.AdaLN(dim)
        self.ada_norm2 = ops.AdaLN(dim)
        self.cross_norm = nn.LayerNorm(dim)
        self.drop_cross = nn.Dropout(dropout_rate)

    def forward(self, x, z, c1=None, c2=None, shift=None, mask=None):

        self_attn_mask = None 
        cross_attn_mask = None 

        if mask is not None:
            
            if len(mask.shape) < len(x.shape):
                self_attn_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
                cross_attn_mask = mask.unsqueeze(-1)
            else:
                self_attn_mask = mask.clone()
                cross_attn_mask = mask[..., :1]

        x = self.ada_norm1(x, c1) if c1 is not None else self.norm1(x) 
        x = x + self.drop_att(self.attention(x, shift=shift, mask=self_attn_mask))
        
        x = self.cross_norm(x)
        x = x + self.drop_cross(self.cross_attention(x, z, cross_attn_mask))
        
        x = self.ada_norm2(x, c2) if c2 is not None else self.norm2(x) 
        x = x + self.drop_mlp(self.submodule(x))

        return x        


