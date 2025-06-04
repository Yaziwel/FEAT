# Copyright (c) Shanghai AI Lab. All rights reserved.
import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat

from rwkv.restormer_arch import FeedForward as ReFFN

T_MAX = 8 * 8

from torch.utils.cpp_extension import load

wkv_cuda = load(name="wkv", sources=["/home/data/whh/Endora/rwkv/cuda/wkv_op.cpp",
                                     "/home/data/whh/Endora/rwkv/cuda/wkv_cuda.cu"],
                verbose=True,
                extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3',
                                   f'-DTmax={T_MAX}'])


class SiLU(nn.Module):
    def __init__(self, inplace=False):
        super(SiLU, self).__init__()
        self.silu = nn.SiLU(inplace)

    def forward(self, x):
        return self.silu(x) / 0.596


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        device = k.device
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device=device, memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y.contiguous()

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        device = k.device
        gw = torch.zeros((B, C), device=device).contiguous()
        gu = torch.zeros((B, C), device=device).contiguous()
        gk = torch.zeros((B, T, C), device=device).contiguous()
        gv = torch.zeros((B, T, C), device=device).contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half().contiguous(), gu.half().contiguous(), gk.half().contiguous(), gv.half().contiguous())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16().contiguous(), gu.bfloat16().contiguous(), gk.bfloat16().contiguous(), gv.bfloat16().contiguous())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw.contiguous(), gu.contiguous(), gk.contiguous(), gv.contiguous())


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.contiguous(), u.contiguous(), k.contiguous(), v.contiguous())


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy',
                 key_norm=False, recurrence=2):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd

        self.dwconv2d = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)
        self.dwconv1d = nn.Conv1d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)

        self.recurrence = recurrence

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
            self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
            self.alpha = nn.Parameter(torch.zeros((2, self.n_embd)))

    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr

        h, w = resolution
        if self.recurrence == 2:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.dwconv2d(x) + x
            x = rearrange(x, 'b c h w -> b (h w) c')
        else:
            b, f, _ = x.size()
            x = rearrange(x, '(b h w) f c -> (b f) c h w', f=f, h=h, w=w)
            x = self.dwconv2d(x) + x
            x = rearrange(x, '(b f) c h w -> (b h w) c f', f=f, h=h, w=w)
            x = self.dwconv1d(x) + x
            x = rearrange(x, 'b c f -> b f c')

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution, res=None):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution)

        if res is not None:
            res_v = res - v
            v = v + res * self.alpha[0]
            res = res_v

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            else:
                h, w = resolution
                k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w)
                v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w)
                v = RUN_CUDA(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                k = rearrange(k, 'b (w h) c -> b (h w) c', h=h, w=w)
                v = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w)

        x = v
        if res is not None:
            x = x + res * self.alpha[1]
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4,
                 init_mode='fancy', key_norm=False, recurrence=2):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, init_mode,
                                    key_norm=key_norm, recurrence=recurrence)

        self.ffn = ReFFN(n_embd, hidden_rate, bias=False, recurrence=recurrence)

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

        self.adaLN_modulation = nn.Sequential(
            SiLU(),
            nn.Linear(n_embd, 6 * n_embd, bias=True)
        )

    def forward(self, x, c, resolution=None, res=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + self.gamma1 * gate_msa.unsqueeze(1) * self.att(modulate(self.ln1(x), shift_msa, scale_msa), resolution, res)
        x = x + self.gamma2 * gate_mlp.unsqueeze(1) * self.ffn(modulate(self.ln2(x), shift_mlp, scale_mlp), resolution)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    import time
    from thop import profile, clever_format

    '''
    !!!!!!!!
    Caution: Please comment out the code related to reparameterization and retain only the 5x5 convolutional layer in the OmniShift.
    !!!!!!!!
    '''

    x = torch.zeros((1, 64, 1152)).type(torch.FloatTensor).cuda()
    c = torch.zeros((1,)).type(torch.FloatTensor).cuda()
    model = RWKV6Block(1152, 64, 64, 0)
    model.cuda()

    since = time.time()
    y = model(x, c)
    print("time", time.time() - since)

    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], '%.6f')
    print('flops', flops)
    print('params', params)
    print(count_parameters(model) / 1e6)
    # print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    # print("Params=", str(params/1e6)+'{}'.format("M"))
