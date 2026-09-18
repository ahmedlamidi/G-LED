"""Differentiable CT operators for PyTorch, in the geometry of common/ct.py
(fan beam, flat detector, DSO 1000, ODD 600, unit pixels and detector bins).

FanFBP       filtered backprojection written in torch: cosine pre-weighting, ramp
             filter (FFT), pixel-driven distance-weighted backprojection with
             linear interpolation. It is linear, and its backward pass is the
             exact transpose (scatter-add), so it can sit inside a network
             (DuDoTrans's consistency layer). torch-radon, which the published
             dual-domain codes use, does not build for current GPUs.
astra_project  the measured-views forward projector A of common/ct.py as an
             autograd function; its backward pass is ASTRA's backprojection
             A^T (DPS's likelihood gradient).

ASTRA's conventions, matched here and checked in the tests: for view angle t the
source is at DSO (sin t, -cos t), the detector axis is (cos t, sin t), image
row 0 is the top (+y) and column 0 the left (-x).
"""
import math

import numpy as np
import torch

from .config import DETECTOR_COUNT, DSO, N_PIX, ODD, SAMPLE_H


class _Backproject(torch.autograd.Function):
    """Weighted pixel-driven backprojection  f = sum_views w(x, y, t) * interp(q_t, u(x, y, t))."""

    @staticmethod
    def forward(ctx, q, geom):
        ctx.geom = geom
        ctx.q_shape = q.shape
        B, V, D = q.shape
        out = q.new_zeros(B, geom.n * geom.n)
        for v0 in range(0, V, geom.chunk):
            i0, frac, w = geom.rays(v0, min(v0 + geom.chunk, V), q.device, q.dtype)     # [v, P]
            qc = q[:, v0:v0 + i0.shape[0]]                                             # [B, v, D]
            a = torch.gather(qc, 2, i0.unsqueeze(0).expand(B, -1, -1))
            b = torch.gather(qc, 2, (i0 + 1).unsqueeze(0).expand(B, -1, -1))
            out += ((a * (1 - frac) + b * frac) * w).sum(dim=1)
        return out.view(B, geom.n, geom.n)

    @staticmethod
    def backward(ctx, grad):
        geom = ctx.geom
        B, V, D = ctx.q_shape
        g = grad.reshape(B, 1, -1)
        gq = grad.new_zeros(B, V, D)
        for v0 in range(0, V, geom.chunk):
            i0, frac, w = geom.rays(v0, min(v0 + geom.chunk, V), grad.device, grad.dtype)
            v = i0.shape[0]
            gw = g * w                                                                 # [B, v, P]
            gq[:, v0:v0 + v].scatter_add_(2, i0.unsqueeze(0).expand(B, -1, -1), gw * (1 - frac))
            gq[:, v0:v0 + v].scatter_add_(2, (i0 + 1).unsqueeze(0).expand(B, -1, -1), gw * frac)
        return gq, None


class FanFBP(torch.nn.Module):
    """Sinogram rows [B, V, D] at `angles` (radians) -> image [B, n, n].

    `weight` is the angular quadrature weight of each view; the default,
    2 pi / V, is the full-scan rule ASTRA's FBP applies to whatever views it is
    given, so a limited arc comes out with the same normalization as the FBP
    baseline's images."""

    def __init__(self, angles, n=N_PIX, n_det=DETECTOR_COUNT, dso=DSO, odd=ODD, weight=None, chunk=24):
        super().__init__()
        self.n, self.n_det, self.dso, self.odd, self.chunk = n, n_det, float(dso), float(odd), chunk
        angles = np.asarray(angles, dtype=np.float64)
        self.register_buffer('angles', torch.tensor(angles, dtype=torch.float32), persistent=False)
        self.weight = (2 * math.pi / len(angles)) if weight is None else float(weight)
        self.mag = (self.dso + self.odd) / self.dso                  # virtual detector (through the origin) -> real one
        a = (torch.arange(n_det, dtype=torch.float32) - (n_det - 1) / 2) / self.mag
        self.register_buffer('preweight', self.dso / torch.sqrt(self.dso ** 2 + a ** 2), persistent=False)
        size = 2 ** int(math.ceil(math.log2(2 * n_det)))
        self.size = size
        # Ram-Lak kernel sampled on the virtual detector (bin spacing da = 1 / mag), Kak & Slaney eq. 3.61:
        # h(0) = 1 / (4 da^2), h(k da) = -1 / (k pi da)^2 for odd k, 0 for even k. Its DFT times da is the
        # filter, and already carries the da of the convolution integral.
        da = 1 / self.mag
        k = torch.arange(size, dtype=torch.float64)
        k = torch.minimum(k, size - k)
        h = torch.where(k % 2 == 1, -1 / (k * math.pi * da) ** 2, torch.zeros_like(k))
        h[0] = 1 / (4 * da * da)
        self.register_buffer('ramp', (torch.fft.fft(h).real * da), persistent=False)
        c = torch.arange(n, dtype=torch.float32) - (n - 1) / 2
        self.register_buffer('px', c.repeat(n), persistent=False)                       # x of every pixel, row-major
        self.register_buffer('py', (-c).repeat_interleave(n), persistent=False)         # y: row 0 is the top

    def rays(self, v0, v1, device, dtype):
        """For views v0..v1 and every pixel: lower detector bin, interpolation weight, backprojection weight."""
        t = self.angles[v0:v1].to(device=device, dtype=dtype).unsqueeze(1)
        px, py = self.px.to(device=device, dtype=dtype), self.py.to(device=device, dtype=dtype)
        along = -px * torch.sin(t) + py * torch.cos(t)               # from the origin away from the source
        U = (self.dso + along) / self.dso
        a = (px * torch.cos(t) + py * torch.sin(t)) / U              # coordinate on the virtual detector
        idx = a * self.mag + (self.n_det - 1) / 2
        inside = (idx >= 0) & (idx <= self.n_det - 1)
        idx = idx.clamp(0, self.n_det - 1 - 1e-4)
        i0 = idx.floor()
        w = inside.to(dtype) / (U * U)
        return i0.long(), idx - i0, w

    def filter(self, sino):
        work = sino.dtype if sino.dtype in (torch.float32, torch.float64) else torch.float32   # no FFT in half precision
        x = (sino * self.preweight.to(sino.dtype)).to(work)
        X = torch.fft.fft(x, n=self.size, dim=-1) * self.ramp.to(work)
        return torch.fft.ifft(X, dim=-1).real[..., :self.n_det].to(sino.dtype)

    def forward(self, sino):
        return 0.5 * self.weight * _Backproject.apply(self.filter(sino), self)


class _AstraProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, op):
        ctx.op = op
        arr = img.detach().float().cpu().numpy()
        out = np.stack([op.fp(a) for a in arr.reshape(-1, *arr.shape[-2:])])
        return torch.from_numpy(out).to(img.device).view(*img.shape[:-2], *out.shape[-2:])

    @staticmethod
    def backward(ctx, grad):
        arr = grad.detach().float().cpu().numpy()
        out = np.stack([ctx.op.bp(a) for a in arr.reshape(-1, *arr.shape[-2:])])
        return torch.from_numpy(out).to(grad.device).view(*grad.shape[:-2], *out.shape[-2:]), None


def astra_project(img, op):
    """A img for a common.ct.MeasuredOperator `op`; img [..., n, n] -> [..., views, detectors]."""
    return _AstraProject.apply(img, op)


def all_angles():
    return np.linspace(0, 2 * np.pi, SAMPLE_H, endpoint=False)
