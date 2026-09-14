"""ASTRA operators in SD-Flow's reconstruction geometry.

full_fbp is train_test_spatial/test_diff.py's _fbp_reconstruct and sparse_fbp is
the FBP inside data/dicom_preprocess.py's compute_fbp_reprojection, call for
call, so the label and the conditioning image are the arrays SD-Flow uses.

The iterative methods fit the measured rows as projections. dicom_dataset maps
each sinogram p to s = 2 (p - min) / (max - min) - 1, and min is normally 0
(rays that only cross air), so y = s + 1 is the projection up to a positive
per-slice scale. FBP is linear, so the label FBP(s) equals FBP(s + 1) - K with
K = FBP(1): an image z fitted to y maps to the label's units as z - K.
"""
import warnings

import astra
import numpy as np

from .config import DETECTOR_COUNT, DSO, N_PIX, ODD, SAMPLE_H

ALL_ANGLES = np.linspace(0, 2 * np.pi, SAMPLE_H, endpoint=False).astype(np.float32)


def _geometry(angles):
    vol_geom = astra.create_vol_geom(N_PIX, N_PIX)
    proj_geom = astra.create_proj_geom('fanflat', 1.0, DETECTOR_COUNT,
                                       np.asarray(angles, dtype=np.float32), DSO, ODD)
    return vol_geom, proj_geom


def _fbp(sino, angles):
    sino = np.ascontiguousarray(sino, dtype=np.float32)
    if not astra.use_cuda():
        return _fbp_cpu_approx(sino, angles)
    vol_geom, proj_geom = _geometry(angles)
    sino_id = astra.data2d.create('-sino', proj_geom, sino)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    alg_id = astra.algorithm.create(cfg)
    try:
        astra.algorithm.run(alg_id)
        return astra.data2d.get(rec_id).astype(np.float32)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sino_id)
        astra.data2d.delete(rec_id)


_warned_cpu = False


def _fbp_cpu_approx(sino, angles):
    """ASTRA has no CPU fan-beam FBP. This ramp-filtered backprojection only lets
    the pipeline run on a machine without a GPU; never report numbers from it."""
    global _warned_cpu
    if not _warned_cpu:
        warnings.warn('ASTRA has no CUDA here: using an approximate CPU FBP (smoke tests only)')
        _warned_cpu = True
    n = sino.shape[1]
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    ramp = np.abs(np.fft.fftfreq(size)).astype(np.float32)
    filtered = np.real(np.fft.ifft(np.fft.fft(sino, size, axis=1) * ramp, axis=1))[:, :n]
    op = MeasuredOperator(angles=angles)
    try:
        return op.bp(filtered.astype(np.float32)) * np.float32(np.pi / len(angles))
    finally:
        op.close()


def full_fbp(sino):
    """FBP of a full 720-view sinogram: SD-Flow's label and evaluation FBP."""
    return _fbp(sino, ALL_ANGLES)


def sparse_fbp(sino, cond):
    """FBP from the measured rows only. On the normalized sinogram this is the
    image behind SD-Flow's third conditioning channel."""
    cond = list(cond)
    return _fbp(np.asarray(sino)[cond, :], ALL_ANGLES[cond])


def fbp_of_ones():
    """K = FBP(1), the offset between FBP(s + 1) and the label FBP(s)."""
    return full_fbp(np.ones((SAMPLE_H, DETECTOR_COUNT), dtype=np.float32))


def neg_laplacian(img):
    """D^T D for forward differences with Neumann boundaries."""
    p = np.pad(img, 1, mode='edge')
    return 4 * img - p[:-2, 1:-1] - p[2:, 1:-1] - p[1:-1, :-2] - p[1:-1, 2:]


class MeasuredOperator:
    """A: 512 x 512 image -> the measured sinogram rows, and its adjoint."""

    def __init__(self, cond=None, angles=None):
        if angles is None:
            angles = ALL_ANGLES[list(cond)]
        self.n_views = len(angles)
        self.vol_geom, self.proj_geom = _geometry(angles)
        kind = 'cuda' if astra.use_cuda() else 'line_fanflat'
        self.proj_id = astra.create_projector(kind, self.proj_geom, self.vol_geom)
        self._norm_sq = None

    def fp(self, img):
        sino_id, sino = astra.create_sino(np.ascontiguousarray(img, dtype=np.float32), self.proj_id)
        astra.data2d.delete(sino_id)
        return sino

    def bp(self, sino):
        vol_id, img = astra.create_backprojection(np.ascontiguousarray(sino, dtype=np.float32), self.proj_id)
        astra.data2d.delete(vol_id)
        return img

    def normal(self, img):
        return self.bp(self.fp(img))

    def norm_sq(self, iters=30):
        """Largest eigenvalue of A^T A, by power iteration."""
        if self._norm_sq is None:
            v = np.random.default_rng(0).standard_normal((N_PIX, N_PIX)).astype(np.float32)
            for _ in range(iters):
                v = self.normal(v)
                self._norm_sq = float(np.linalg.norm(v))
                v /= self._norm_sq
        return self._norm_sq

    @staticmethod
    def cg(apply, rhs, x0=None, iters=50, rtol=1e-6):
        """Conjugate gradients for apply(x) = rhs, apply symmetric positive definite."""
        x = np.zeros_like(rhs) if x0 is None else x0.astype(np.float32, copy=True)
        r = rhs - apply(x)
        p = r.copy()
        rr = rr0 = float(np.vdot(r, r))
        for _ in range(iters):
            if rr == 0 or rr <= rtol ** 2 * rr0:
                break
            ap = apply(p)
            alpha = rr / float(np.vdot(p, ap))
            x += alpha * p
            r -= alpha * ap
            rr_new = float(np.vdot(r, r))
            p = r + (rr_new / rr) * p
            rr = rr_new
        return x

    def prox(self, z0, y, lam, iters=100):
        """argmin_z ||A z - y||^2 + lam ||z - z0||^2 (DOLCE eq. 10 with lam = 1 / gamma)."""
        return self.cg(lambda v: self.normal(v) + lam * v, self.bp(y) + lam * z0, x0=z0, iters=iters)

    def rls(self, y, beta, iters=50):
        """argmin_z ||A z - y||^2 + beta ||grad z||^2: regularized least squares."""
        return self.cg(lambda v: self.normal(v) + beta * neg_laplacian(v), self.bp(y), iters=iters)

    def sart(self, y, iters, min_constraint=0.0):
        """ASTRA SART. One ASTRA iteration is an update from one view."""
        cuda = astra.use_cuda()
        sino_id = astra.data2d.create('-sino', self.proj_geom, np.ascontiguousarray(y, dtype=np.float32))
        rec_id = astra.data2d.create('-vol', self.vol_geom, 0.0)
        cfg = astra.astra_dict('SART_CUDA' if cuda else 'SART')
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = rec_id
        if not cuda:
            cfg['ProjectorId'] = self.proj_id
        if min_constraint is not None:
            cfg['option'] = {'MinConstraint': float(min_constraint)}
        alg_id = astra.algorithm.create(cfg)
        try:
            astra.algorithm.run(alg_id, int(iters))
            return astra.data2d.get(rec_id).astype(np.float32)
        finally:
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(sino_id)
            astra.data2d.delete(rec_id)

    def close(self):
        astra.projector.delete(self.proj_id)
