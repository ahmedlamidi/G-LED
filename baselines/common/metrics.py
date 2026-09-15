"""Metrics, computed the same way for every method.

ct_scores reports three kinds of SSIM / PSNR:
* body and HU: both images in HU with the label's calibration (air is 0 in
  label + K, the label's soft-tissue peak is 0 HU), clipped to [-1000, 1000] HU
  (data range 2000). 'body' averages over the patient only (label above
  -500 HU inside the field of view), 'hu' over the whole image. The calibration
  comes from the label alone, so every method is scored on the same scale.
* legacy (ct_metrics): validation/compare_ssim.py, which SD-Flow reported
  before: both images normalized by the label's min/max, data range 1, whole
  image. The FBP rim at the edge of the field of view sets the label's minimum
  and squeezes the body into the top quarter of that range, so these scores
  are high for every method, even a body outline with nothing inside.
"""
import numpy as np
from skimage.filters import threshold_otsu
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

HU_RANGE = (-1000.0, 1000.0)
# (key, column title), in the order the tables show them
METRICS = [('ssim_body', 'SSIM body'), ('psnr_body', 'PSNR body (dB)'),
           ('ssim_hu', 'SSIM HU'), ('psnr_hu', 'PSNR HU (dB)'),
           ('ssim_legacy', 'SSIM legacy'), ('psnr_legacy', 'PSNR legacy (dB)')]


def ct_metrics(pred, label):
    """SSIM / PSNR / MSE the way validation/compare_ssim.py scores SD-Flow's CT
    images: both images normalized by the label's min/max, data_range 1."""
    pred = np.asarray(pred, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32)
    lo, hi = label.min(), label.max()
    if hi > lo:
        pred = (pred - lo) / (hi - lo)
        label = (label - lo) / (hi - lo)
    return {'ssim': float(structural_similarity(label, pred, data_range=1.0)),
            'psnr': float(peak_signal_noise_ratio(label, pred, data_range=1.0)),
            'mse': float(np.mean((label - pred) ** 2))}


def _fov(n, frac):
    yy, xx = np.mgrid[:n, :n]
    return np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2) < frac * n


def water_level(mu):
    """Soft-tissue level of an image proportional to mu with air at 0 (label + K).

    Each sinogram is rescaled on its own, so the images have no fixed HU scale.
    The most common value above the Otsu air/body threshold inside the field of
    view is taken as water (0 HU).
    """
    fov = mu[_fov(mu.shape[0], 0.45)]
    body = fov[fov > threshold_otsu(fov)]
    hist, edges = np.histogram(body, bins=200)
    k = int(np.convolve(hist, np.ones(5) / 5, mode='same').argmax())
    return 0.5 * (edges[k] + edges[k + 1])


def to_hu(mu, water):
    """HU of an image in label + K units (air at 0) whose water level is `water`."""
    return 1000 * (mu / water - 1)


def body_mask(hu):
    """The patient: above -500 HU inside the field of view."""
    return (hu > -500) & _fov(hu.shape[0], 0.47)


def ct_scores(pred, label, offset, water=None):
    """Body, HU and legacy SSIM / PSNR of pred against label (both in label
    units); offset is K = FBP(1), water the label's level (computed if None)."""
    pred = np.asarray(pred, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32)
    if water is None:
        water = water_level(label + offset)
    lo, hi = HU_RANGE
    hl = np.clip(to_hu(label + offset, water), lo, hi)
    hp = np.clip(to_hu(pred + offset, water), lo, hi)
    ssim_hu, ssim_map = structural_similarity(hl, hp, data_range=hi - lo, full=True)
    body = body_mask(hl)
    mse_body = float(np.mean((hl - hp)[body] ** 2))
    legacy = ct_metrics(pred, label)
    return {'ssim_body': float(ssim_map[body].mean()),
            'psnr_body': float(10 * np.log10((hi - lo) ** 2 / mse_body)) if mse_body > 0 else float('inf'),
            'ssim_hu': float(ssim_hu),
            'psnr_hu': float(peak_signal_noise_ratio(hl, hp, data_range=hi - lo)),
            'ssim_legacy': legacy['ssim'],
            'psnr_legacy': legacy['psnr']}
