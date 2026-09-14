"""DOLCE pieces shared by training and sampling (Liu et al., ICCV 2023)."""
import numpy as np

from ..common.data import minmax_unit
from ..common.unet import DiffusionUNet


def build_model(cfg):
    # The noisy image and the condition are concatenated on channels (paper eq. 7)
    return DiffusionUNet(in_ch=2, out_ch=1, base=cfg['base'], ch_mult=tuple(cfg['ch_mult']),
                         num_res=cfg['num_res'], attn_levels=tuple(cfg['attn_levels']),
                         dropout=cfg['dropout'], emb='sinusoidal')


def alphas_cumprod(T):
    """guided-diffusion's 'linear' schedule, which DOLCE uses with T = 2000."""
    scale = 1000 / T
    betas = np.linspace(scale * 1e-4, scale * 0.02, T, dtype=np.float64)
    return np.cumprod(1.0 - betas)


def condition(rls_img):
    """DOLCE's condition: the RLS image, min-max normalized per slice (to [-1, 1] here)."""
    return minmax_unit(rls_img)
