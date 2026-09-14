"""The metrics SD-Flow reports, computed the same way for every method."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


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
