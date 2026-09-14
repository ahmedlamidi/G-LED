"""SART baseline: ASTRA's SART on the measured views, with non-negativity.

The number of sweeps over the measured views is picked on the validation
patient by SSIM against the label.

    python -m baselines.sart.reconstruct
"""
from ..common import classical
from ..common.ct import MeasuredOperator

SWEEPS = (5, 10, 20, 50, 100, 200)


def main():
    args = classical.build_parser(__doc__).parse_args()
    cond, offset, _ = classical.context(args)
    op = MeasuredOperator(cond)

    def recon(sino, sweeps):
        return op.sart(sino[cond] + 1.0, iters=int(sweeps) * len(cond), min_constraint=0.0) - offset

    classical.run(args, 'sart', recon, SWEEPS, 'sweeps')
    op.close()


if __name__ == '__main__':
    main()
