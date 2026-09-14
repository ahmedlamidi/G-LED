"""Time-conditioned U-Net for the DOLCE and SWORD reimplementations.

It follows the guided-diffusion U-Net that DOLCE builds on (residual blocks with
scale-shift time conditioning, self-attention at low resolution), with the
1/sqrt(2)-rescaled skips that both DOLCE and NCSN++ use. The time input is a
diffusion step (sinusoidal embedding, DOLCE) or log(sigma) (random Fourier
features, SWORD's NCSN++).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch):
    return nn.GroupNorm(math.gcd(32, ch), ch)


def _zero(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def sinusoidal_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FourierEmbedding(nn.Module):
    """Random Fourier features, as NCSN++ embeds log(sigma) (scale 16)."""

    def __init__(self, dim, scale=16.0):
        super().__init__()
        self.register_buffer('W', torch.randn(dim // 2) * scale)

    def forward(self, x):
        proj = x.float()[:, None] * self.W[None] * 2 * math.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout):
        super().__init__()
        self.in_layers = nn.Sequential(_norm(cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1))
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * cout))
        self.out_norm = _norm(cout)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout),
                                        _zero(nn.Conv2d(cout, cout, 3, padding=1)))
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.out_layers(self.out_norm(h) * (1 + scale) + shift)
        return (self.skip(x) + h) / math.sqrt(2)


class Attention(nn.Module):
    def __init__(self, ch, head_ch=64):
        super().__init__()
        self.heads = max(ch // head_ch, 1)
        self.norm = _norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = _zero(nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv.transpose(-1, -2).unbind(1)          # each (b, heads, hw, d)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return (x + self.proj(out)) / math.sqrt(2)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class DiffusionUNet(nn.Module):
    """Input height and width must divide by 2 ** (len(ch_mult) - 1)."""

    def __init__(self, in_ch, out_ch, base=64, ch_mult=(1, 1, 2, 2, 4, 4), num_res=2,
                 attn_levels=(4, 5), dropout=0.0, emb='sinusoidal'):
        super().__init__()
        self.base = base
        self.emb_kind = emb
        self.down_factor = 2 ** (len(ch_mult) - 1)
        emb_dim = 4 * base
        if emb == 'fourier':
            self.fourier = FourierEmbedding(base)
        self.emb_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.inp = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down = nn.ModuleList()
        skips, ch = [base], base
        for level, mult in enumerate(ch_mult):
            for _ in range(num_res):
                blocks = [ResBlock(ch, base * mult, emb_dim, dropout)]
                ch = base * mult
                if level in attn_levels:
                    blocks.append(Attention(ch))
                self.down.append(nn.ModuleList(blocks))
                skips.append(ch)
            if level != len(ch_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                skips.append(ch)

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, dropout), Attention(ch),
                                  ResBlock(ch, ch, emb_dim, dropout)])

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mult))):
            for i in range(num_res + 1):
                blocks = [ResBlock(ch + skips.pop(), base * mult, emb_dim, dropout)]
                ch = base * mult
                if level in attn_levels:
                    blocks.append(Attention(ch))
                if level != 0 and i == num_res:
                    blocks.append(Upsample(ch))
                self.up.append(nn.ModuleList(blocks))

        self.out = nn.Sequential(_norm(ch), nn.SiLU(), _zero(nn.Conv2d(ch, out_ch, 3, padding=1)))

    @staticmethod
    def _run(blocks, h, emb):
        for m in blocks:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        return h

    def forward(self, x, t):
        if x.shape[-2] % self.down_factor or x.shape[-1] % self.down_factor:
            raise ValueError(f'input {tuple(x.shape[-2:])} does not divide by {self.down_factor}')
        emb = self.fourier(t) if self.emb_kind == 'fourier' else sinusoidal_embedding(t, self.base)
        emb = self.emb_mlp(emb)
        h = self.inp(x)
        hs = [h]
        for blocks in self.down:
            h = self._run(blocks, h, emb)
            hs.append(h)
        h = self._run(self.mid, h, emb)
        for blocks in self.up:
            h = self._run(blocks, torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(h)
