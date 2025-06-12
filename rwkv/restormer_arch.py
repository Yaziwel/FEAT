## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, recurrence=2):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.recurrence = recurrence

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, resolution=None):
        h, w = resolution
        if self.recurrence == 1:
            b, f, _ = x.size()
            x = rearrange(x, '(b h w) f c -> (b f) c h w', f=f, h=h, w=w)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        if self.recurrence == 1:
            x = rearrange(x, '(b f) c h w -> (b h w) f c', f=f, h=h, w=w)
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        with torch.no_grad():
            self.alpha = nn.Parameter(torch.zeros(2 * dim))

    def forward(self, x, resolution=None, res=None):
        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if res is not None:
            res = rearrange(res, 'b (h w) c -> b c h w', h=h, w=w)
            res_v = res - v
            v = v + res * self.alpha[0]
            res = res_v

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        if res is not None:
            out = out + res * self.alpha[1]
        out = self.project_out(out)
        out = rearrange(out, 'b c h w -> b (h w) c')
        return out


##########################################################################


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlockControl(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlockControl, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, c, resolution=None, res=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), resolution, res)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp), resolution)
        return x

