"""DuDoTrans (Wang et al., 2021, arXiv:2111.10790): dual-domain transformer.

    Y  --SRT-->  Y~  --FBP-->  X~2 \\
    Y  ----------------FBP-->  X~1 --> RIRM --> X~ = RIRM([X~1, X~2]) + X~1

SRT (sinogram restoration transformer, paper eq. 3): a conv for shallow
features, m = 3 residual blocks of n = 1 Swin transformer module plus a conv,
a conv with a skip from the shallow features, a reconstruction conv, and
Y~ = Y + that. RIRM (eqs. 7-9): shallow conv on [X~1, X~2], depth = 2
sub-modules of width = 4 Swin blocks, a reconstruction conv, plus X~1.
The consistency layer between them is common/torch_ct.FanFBP.

What the paper does not give (the official repository is an empty placeholder)
and is chosen here: embedding dim 48, 4 heads, window 8, MLP ratio 4, one Swin
"module" = a regular plus a shifted-window block, and a stride-2 embedding
with a pixel-shuffle back (the sinogram here is 720 x 816, far larger than the
paper's) so attention runs on 360 x 408 and 256 x 256 tokens. About 0.5 M
parameters, the paper's 0.44 M.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window, shift, mlp_ratio):
        super().__init__()
        self.dim, self.heads, self.window, self.shift = dim, heads, window, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        w = window
        self.bias = nn.Parameter(torch.zeros((2 * w - 1) ** 2, heads))
        nn.init.trunc_normal_(self.bias, std=0.02)
        c = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(w), indexing='ij')).flatten(1)
        rel = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + (w - 1)
        self.register_buffer('rel_index', rel[..., 0] * (2 * w - 1) + rel[..., 1], persistent=False)
        self._masks = {}

    def shift_mask(self, H, W, device):
        """Windows that wrap around after the cyclic shift must not attend across the seam."""
        key = (H, W, str(device))
        if key not in self._masks:
            w, s = self.window, self.shift
            img = torch.zeros(1, H, W, device=device)
            n = 0
            for hs in (slice(0, -w), slice(-w, -s), slice(-s, None)):
                for ws in (slice(0, -w), slice(-w, -s), slice(-s, None)):
                    img[:, hs, ws] = n
                    n += 1
            win = img.view(1, H // w, w, W // w, w).permute(0, 1, 3, 2, 4).reshape(-1, w * w)
            diff = win[:, None, :] - win[:, :, None]
            self._masks[key] = torch.zeros_like(diff).masked_fill(diff != 0, float('-inf'))   # [nW, N, N]
        return self._masks[key]

    def forward(self, x):                                   # x [B, H, W, C], H and W multiples of the window
        B, H, W, C = x.shape
        w, s = self.window, self.shift
        h = self.norm1(x)
        if s:
            h = torch.roll(h, (-s, -s), (1, 2))
        h = h.view(B, H // w, w, W // w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, -1, w * w, C)   # [B, nW, N, C]
        q, k, v = self.qkv(h).view(B, h.shape[1], w * w, 3, self.heads, C // self.heads).permute(3, 0, 1, 4, 2, 5)
        bias = self.bias[self.rel_index].permute(2, 0, 1)[None, None]                                # [1, 1, heads, N, N]
        if s:
            bias = bias + self.shift_mask(H, W, x.device)[None, :, None]
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))                      # [B, nW, heads, N, d]
        h = self.proj(h.transpose(2, 3).reshape(B, -1, w * w, C))
        h = h.view(B, H // w, W // w, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        if s:
            h = torch.roll(h, (s, s), (1, 2))
        x = x + h
        return x + self.mlp(self.norm2(x))


class SwinGroup(nn.Module):
    """`blocks` Swin blocks (alternating regular / shifted windows), a 3x3 conv, and a skip (RSTB-style)."""

    def __init__(self, dim, heads, window, blocks, mlp_ratio, grad_checkpoint):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, heads, window, (window // 2) * (i % 2), mlp_ratio)
                                     for i in range(blocks)])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.grad_checkpoint = grad_checkpoint

    def body(self, x):
        h = x.permute(0, 2, 3, 1)
        for blk in self.blocks:
            h = blk(h)
        return self.conv(h.permute(0, 3, 1, 2)) + x

    def forward(self, x):
        if self.grad_checkpoint and x.requires_grad:
            return checkpoint(self.body, x, use_reentrant=False)
        return self.body(x)


class SwinNet(nn.Module):
    """Shallow conv (stride `patch`) -> Swin groups -> conv + skip -> pixel-shuffle reconstruction conv."""

    def __init__(self, in_ch, dim, heads, window, groups, blocks, mlp_ratio, patch, grad_checkpoint):
        super().__init__()
        self.window, self.patch = window, patch
        self.shallow = nn.Conv2d(in_ch, dim, 3, stride=patch, padding=1)
        self.groups = nn.ModuleList([SwinGroup(dim, heads, window, blocks, mlp_ratio, grad_checkpoint)
                                     for _ in range(groups)])
        self.fuse = nn.Conv2d(dim, dim, 3, padding=1)
        self.recon = nn.Sequential(nn.Conv2d(dim, patch * patch, 3, padding=1), nn.PixelShuffle(patch))

    def forward(self, x):
        H, W = x.shape[-2:]
        m = self.window * self.patch
        x = F.pad(x, (0, (-W) % m, 0, (-H) % m))
        f0 = self.shallow(x)
        f = f0
        for g in self.groups:
            f = g(f)
        return self.recon(self.fuse(f) + f0)[..., :H, :W]


class DuDoTrans(nn.Module):
    def __init__(self, fbp_full, fbp_measured, fill, cond, img_scale, dim=48, heads=4, window=8, mlp_ratio=4.0,
                 srt_groups=3, srt_blocks=2, rirm_depth=2, rirm_width=4, patch=2, data_consistency=True,
                 grad_checkpoint=False):
        """fbp_full / fbp_measured: FanFBP over all views / the measured views. fill [views, n_meas]: the
        matrix that spreads the measured rows over the full sinogram (see train.fill_matrix)."""
        super().__init__()
        self.fbp_full, self.fbp_measured = fbp_full, fbp_measured
        self.register_buffer('fill', fill, persistent=False)
        mask = torch.zeros(fill.shape[0])
        mask[cond] = 1
        self.register_buffer('mask', mask.view(1, 1, -1, 1), persistent=False)
        self.cond, self.img_scale, self.data_consistency = list(cond), float(img_scale), data_consistency
        self.srt = SwinNet(2, dim, heads, window, srt_groups, srt_blocks, mlp_ratio, patch, grad_checkpoint)
        self.rirm = SwinNet(2, dim, heads, window, rirm_depth, rirm_width, mlp_ratio, patch, grad_checkpoint)

    def forward(self, y_meas):
        """y_meas [B, n_meas, det], the measured rows of y = s + 1. Returns the restored sinogram
        [B, views, det] and the images X~1, X~2, X~ [B, n, n] scaled by img_scale."""
        y_in = torch.einsum('vm,bmd->bvd', self.fill, y_meas).unsqueeze(1)                 # [B, 1, views, det]
        y_res = y_in + self.srt(torch.cat([y_in, self.mask.expand_as(y_in)], dim=1))
        if self.data_consistency:
            y_res = y_res * (1 - self.mask) + y_in * self.mask                                # measured rows are kept
        with torch.no_grad():
            x1 = self.fbp_measured(y_meas.float()) * self.img_scale
        x2 = self.fbp_full(y_res[:, 0].float()) * self.img_scale
        x = self.rirm(torch.stack([x1, x2], dim=1).to(y_res.dtype))[:, 0].float() + x1
        return y_res[:, 0], x1, x2, x
