"""Settings every baseline shares with SD-Flow.

Everything here mirrors the SD-Flow code, so all methods see the same measured
rows of the same sinograms and are scored against the same label:

  sinograms       data/dicom_preprocess.py, dicom_dataset: 720 views x 816
                  detectors over 360 deg, each normalized to [-1, 1]
  measured rows   main_diff_bfs.py, cond_indices
  label           train_test_spatial/test_diff.py, _fbp_reconstruct: the FBP of
                  the full sinogram, the image SD-Flow is scored against
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_H = 720                  # views over 360 deg, so one row is 0.5 deg
ANGLE_STEP = 360 / SAMPLE_H
DETECTOR_COUNT = 816
DSO = 1000
ODD = 600
N_PIX = 512

DEFAULT_CACHE_ROOT = os.path.join('data', 'baselines_cache')
DEFAULT_OUT_ROOT = os.path.join('output', 'baselines')


def cond_indices(angle_start, angle_end, angle_stride):
    """Measured sinogram rows, with the same arithmetic as main_diff_bfs.py."""
    start_idx = int(angle_start / ANGLE_STEP)
    end_idx = int(angle_end / ANGLE_STEP)
    return list(range(start_idx, end_idx, angle_stride))


def add_common_args(parser):
    group = parser.add_argument_group('setting shared by every baseline')
    group.add_argument('--angle_start', type=float, default=0)
    group.add_argument('--angle_end', type=float, default=45)
    group.add_argument('--angle_stride', type=int, default=10)
    group.add_argument('--cache_root', default=DEFAULT_CACHE_ROOT,
                       help='where prepare_data.py writes sinograms and derived images')
    group.add_argument('--out_root', default=DEFAULT_OUT_ROOT,
                       help='where checkpoints and reconstructions go')
    group.add_argument('--device', default='cuda:0')
    group.add_argument('--seed', type=int, default=0)
    return parser


def setting_tag(args):
    return f'limited{args.angle_start:g}-{args.angle_end:g}_stride{args.angle_stride}'


def setting_dir(args):
    """prepare_data.py's output for this angle setting."""
    return os.path.join(args.cache_root, setting_tag(args))


def method_dir(args, method):
    path = os.path.join(args.out_root, setting_tag(args), method)
    os.makedirs(path, exist_ok=True)
    return path


def dps_prior(base_dir):
    """DPS's prior under a setting's output folder: the dedicated unconditional model (dps_prior, trained
    with --p_uncond 1) once its training has finished, else DOLCE's unconditional branch."""
    own = os.path.join(base_dir, 'dps_prior')
    if os.path.exists(os.path.join(own, 'train_done')):
        return os.path.join(own, 'last.pt')
    return os.path.join(base_dir, 'dolce', 'last.pt')
