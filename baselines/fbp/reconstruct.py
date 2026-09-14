"""FBP baseline: FBP of the measured views, in the label's units.

The measured rows are used as projections, y = s + 1, and the result is shifted
by K = FBP(1) into the label's units (see common/ct.py). No free parameter.

    python -m baselines.fbp.reconstruct
"""
from ..common import classical
from ..common.ct import sparse_fbp


def main():
    args = classical.build_parser(__doc__).parse_args()
    cond, offset, _ = classical.context(args)
    classical.run(args, 'fbp', lambda sino, _: sparse_fbp(sino + 1.0, cond) - offset)


if __name__ == '__main__':
    main()
