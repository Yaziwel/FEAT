# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
# from scipy.special import inputs
from timm.models.vision_transformer import Mlp, PatchEmbed

import os
import sys

sys.path.append(os.path.split(sys.path[0])[0])

from rwkv.Restore_RWKV import RWKV6Block, RWKV6STBlock, RWKV6FPSBlock, RWKV6ModBlock, Conv3DMod
from rwkv.Restore_RWKV import Block as RWKVBlock
from rwkv.Restore_RWKV import BlockMod as RWKVModBlock
from rwkv.Restore_RWKV import BlockAddMod as RWKVAddModBlock
from rwkv.restormer_arch import TransformerBlockControl as ReBlock


class SiLU(nn.Module):
    def __init__(self, inplace=False):
        super(SiLU, self).__init__()
        self.silu = nn.SiLU(inplace)

    def forward(self, x):
        return self.silu(x) / 0.596


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps                                  #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            # nn.SiLU(),
            SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These  be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core EnDora Model                                #
#################################################################################


class FinalLayer(nn.Module):
    """
    The final layer of FEAT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FEAT(nn.Module):
    """
    Diffusion model with a RWKV backbone.
    """

    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            num_frames=16,
            num_classes=1,
            learn_sigma=True,
            extras=2,
            attention_mode='math',
            skip_connect=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.extras = extras
        self.num_frames = num_frames

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            RWKVBlock(hidden_size, depth, layer_id=i, hidden_rate=mlp_ratio, recurrence=-(i % 2) + 2) for i in
            range(depth)
        ])
        self.restormers = nn.ModuleList([
            ReBlock(hidden_size, num_heads, mlp_ratio, False) for _ in range(depth // 2)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.skip_connect = skip_connect
        if skip_connect:
            self.res_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in EnDora blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(
            self,
            x,
            t,
            attentions=None,
            special_list=[],
            mode="type0",
            y=None,
            use_fp16=False,
            y_image=None,
            use_image_num=0
    ):
        """
        Forward pass of EnDora.
        x: (N, F, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        y_image: tensor of video frames
        use_image_num: how many video frames are used
        """
        if use_fp16:
            x = x.to(dtype=torch.float16)

        batches, frames, channels, high, weight = x.shape
        resolution = (high // self.patch_size, weight // self.patch_size)
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        if self.skip_connect:
            skip_res = self.res_conv(x)
        x = self.x_embedder(x)
        _, t, _ = x.shape
        t = self.t_embedder(t, use_fp16=use_fp16)
        timestep_spatial = repeat(t, 'n d -> (n c) d', c=frames)
        timestep_temp = repeat(t, 'n d -> (n c) d', c=t)
        timestep_channel = repeat(t, 'n d -> (n c) d', c=frames)

        res = x
        res_video = rearrange(res, '(b f) t d -> (b t) f d', b=batches)[:, :(frames - use_image_num), :]

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i + 2]
            re_block = self.restormers[i // 2]

            c = timestep_spatial

            x = spatial_block(x, c, resolution, res)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            x_video = x[:, :(frames - use_image_num), :]
            x_image = x[:, (frames - use_image_num):, :]

            c = timestep_temp

            x_video = temp_block(x_video, c, resolution, res_video)
            x = torch.cat([x_video, x_image], dim=1)

            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)

            c = timestep_channel

            x = re_block(x, c, resolution)

        c = timestep_spatial
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.skip_connect:
            x = x + skip_res
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        # print(x.shape)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   FEAT Configs                                  #
#################################################################################


def FEAT_S(**kwargs):
    return FEAT(depth=18, hidden_size=512, patch_size=2, num_heads=8, **kwargs)


def FEAT_L(**kwargs):
    return FEAT(depth=18, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


FEAT_models = {
    "FEAT-S": FEAT_S,
    "FEAT-L": FEAT_L
}


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    total = np.sum([p.numel() for p in model.parameters()]).item()
    print('  + Number of params: %.2fM' % (total / 1e6))


if __name__ == '__main__':
    import torch
    from thop import profile
    # from torchstat import stat
    from fvcore.nn import FlopCountAnalysis, parameter_count

    torch.cuda.set_device(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_image_num = 0
    input_size = [16]
    for in_size in input_size:
        print(f'input_size is :{in_size}')
        x = torch.randn(1, 16, 4, in_size, in_size).cuda()
        t = torch.randn(1, ).cuda()
        # network = Endora_S_2_with_rwkv4_add_mod(input_size=16).to(device)
        # network = Endora_S_2_with_rwkv4_mod(input_size=16).to(device)
        # network = Endora_S_2_with_rwkv(input_size=16).to(device)
        # y = network(x, t, use_image_num=use_image_num)
        # print(y.shape)
        # network = EnDora_XL_2(input_size=in_size).to(device)
        # network = Endora_S_2_with_rwkv_no_patch(input_size=16).to(device)
        # network = Endora_S_2_with_rwkv_dc(input_size=16).to(device)
        # network = Endora_S_2_with_rwkv(input_size=16).to(device)
        # network = Endora_L_2_with_rwkv(input_size=16).to(device)
        # network = Endora_B_2_with_rwkv(input_size=in_size).to(device)
        network = EnDora_S_2(input_size=in_size).to(device)
        # network = Endora_S_2_with_rwkv_st().to(device)

        flops, params = profile(network, inputs=(x, t))
        print(f'flops:{(flops / 1e9):.4f} GFLOPs, params:{(params / 1e6):.4f} MParams')
    # print(stat(network, (16, 4, 16, 16)))

    # print_model_parm_nums(network)
    #
    # # 使用 fvcore 分析 FLOPs 和参数量
    # flops = FlopCountAnalysis(network, (x, t))  # 将多个输入作为元组传递
    # params = parameter_count(network)
    #
    # print(f"FLOPs: {flops.total() / 1e9} G")
    # print(f"Params: {params[''] / 1e6} M")