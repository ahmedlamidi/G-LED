"""TV baseline: min_z 1/2 ||A z - y||^2 + lambda TV(z), z >= 0, on the measured views.

Solved with FISTA and a Chambolle TV prox (Beck & Teboulle 2009). This solves
the regularized inverse problem; it is not FBP followed by TV denoising, which
is what dicom_fbp_degraded.py calls TV. The prox weight, relative to the
typical image intensity, is picked on the validation patient.

    python -m baselines.tv.reconstruct
"""
import numpy as np
from skimage.restoration import denoise_tv_chambolle

from ..common import classical
from ..common.config import N_PIX
from ..common.ct import MeasuredOperator

WEIGHTS = (0.001, 0.003, 0.01, 0.03, 0.1)


def tv_fista(op, y, weight, iters=100):
    step = 1.0 / op.norm_sq()
    x = np.zeros((N_PIX, N_PIX), np.float32)
    s, t = x.copy(), 1.0
    for _ in range(iters):
        v = s - step * op.bp(op.fp(s) - y)
        x_next = np.maximum(denoise_tv_chambolle(v, weight=weight, max_num_iter=50), 0).astype(np.float32)
        t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
        s = x_next + ((t - 1) / t_next) * (x_next - x)
        x, t = x_next, t_next
    return x


def main():
    parser = classical.build_parser(__doc__)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()
    cond, offset, stats = classical.context(args)
    op = MeasuredOperator(cond)
    scale = stats['z_scale']

    def recon(sino, weight):
        return tv_fista(op, sino[cond] + 1.0, weight * scale, args.iters) - offset

    classical.run(args, 'tv', recon, WEIGHTS, 'weight')
    op.close()


if __name__ == '__main__':
    main()
