"""SWORD pieces shared by training and sampling (Xu et al., IEEE TMI 2024).

SWORD models the sinogram in the Haar wavelet domain with two score models (VE
SDE, NCSN++-style networks): a full-frequency model on all four bands and a
high-frequency model on the three detail bands. It works on y = s + 1, where air
is 0, zero-padded from 720 x 816 to 768 x 832 so the 384 x 416 bands halve
cleanly through the network.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..common.config import DETECTOR_COUNT, SAMPLE_H
from ..common.runtime import autocast
from ..common.unet import DiffusionUNet

PAD_H, PAD_W = 768, 832
TOP, LEFT = (PAD_H - SAMPLE_H) // 2, (PAD_W - DETECTOR_COUNT) // 2
BANDS = {'full': 4, 'high': 3}


def pad(y):
    return F.pad(y, (LEFT, PAD_W - DETECTOR_COUNT - LEFT, TOP, PAD_H - SAMPLE_H - TOP))


def unpad(p):
    return p[..., TOP:TOP + SAMPLE_H, LEFT:LEFT + DETECTOR_COUNT]


def dwt(x):
    """Orthonormal 2D Haar: (B, 1, H, W) -> (B, 4, H/2, W/2) as LL, LH, HL, HH."""
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return torch.cat([a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d], dim=1) / 2


def iwt(w):
    ll, lh, hl, hh = w[:, 0:1], w[:, 1:2], w[:, 2:3], w[:, 3:4]
    out = torch.empty(w.shape[0], 1, 2 * w.shape[2], 2 * w.shape[3], dtype=w.dtype, device=w.device)
    out[..., 0::2, 0::2] = (ll + lh + hl + hh) / 2
    out[..., 0::2, 1::2] = (ll - lh + hl - hh) / 2
    out[..., 1::2, 0::2] = (ll + lh - hl - hh) / 2
    out[..., 1::2, 1::2] = (ll - lh - hl + hh) / 2
    return out


def to_bands(sino, band):
    """(B, 1, 720, 816) normalized sinograms -> the model's wavelet bands of y = s + 1."""
    w = dwt(pad(sino + 1.0))
    return w if band == 'full' else w[:, 1:]


def build_model(cfg):
    n = BANDS[cfg['band']]
    return DiffusionUNet(in_ch=n, out_ch=n, base=cfg['base'], ch_mult=tuple(cfg['ch_mult']),
                         num_res=cfg['num_res'], attn_levels=tuple(cfg['attn_levels']),
                         dropout=0.0, emb='fourier')


def score(model, x, sigma, amp=True):
    """Score at noise level sigma (B,): the network output divided by sigma, as
    NCSN++ does with scale_by_sigma; the network sees log(sigma)."""
    with autocast(x.device, amp):
        out = model(x, torch.log(sigma))
    return out.float() / sigma.view(-1, 1, 1, 1)


def sigma_schedule(cfg, n_levels):
    """Discrete VE noise levels, from sigma_max down to sigma_min."""
    return np.exp(np.linspace(math.log(cfg['sigma_max']), math.log(cfg['sigma_min']), n_levels))
