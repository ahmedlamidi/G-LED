"""Build the data every baseline reads.

Patients are the folders under --data_root in natural sort order: the first 7
train, the next 1 validation, the last 2 test. For each split this writes, under
<cache_root>/<setting>/<split>/:

  slices.json   one entry per slice: id, patient, series, slice, sinogram path
  label.npy     FBP of the full sinogram, the image SD-Flow is scored against
  fbp_in.npy    FBP of the measured rows, the image behind SD-Flow's conditioning
  rls.npy       regularized least squares from the measured rows (DOLCE's condition)

and, once per setting, split.json, stats.json and K.npy (FBP of an all-ones
sinogram, see common/ct.py). Sinograms are made by SD-Flow's own dicom_dataset,
so every method starts from exactly SD-Flow's input.

    python -m baselines.prepare_data --data_root data
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
from tqdm import tqdm

from .common import splits
from .common.config import (ANGLE_STEP, DETECTOR_COUNT, N_PIX, add_common_args,
                            cond_indices, setting_dir)
from .common.ct import MeasuredOperator, fbp_of_ones, full_fbp, sparse_fbp
from .common.data import SplitData, sdflow_dicom_dataset

SPLITS = ('train', 'val', 'test')
IMAGES = ('label', 'fbp_in', 'rls')


def _cache_series(job):
    """Run SD-Flow's dicom_dataset on one series; it caches every sinogram."""
    series, sino_cache = job
    dicom_dataset = sdflow_dicom_dataset()
    ds = dicom_dataset(data_path=series, detector_count=DETECTOR_COUNT,
                       angle_step=ANGLE_STEP, cache_dir=sino_cache)
    return series, list(ds.index_map)


def _manifest(split, index_maps, test_stride):
    manifest = {}
    for name in SPLITS:
        slices = []
        stride = test_stride if name == 'test' else 1
        for patient in split[name]:
            for series in splits.series_dirs(patient):
                paths = index_maps[series]
                for j in range(0, len(paths), stride):
                    slices.append({'id': f'{name}_{len(slices):05d}',
                                   'patient': os.path.basename(os.path.normpath(patient)),
                                   'series': os.path.relpath(series, patient),
                                   'slice': j,
                                   'sino': paths[j]})
        manifest[name] = slices
    return manifest


def _write_split(out_dir, slices, cond, op, rls_beta, rls_iters, force):
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, 'DONE')
    manifest_path = os.path.join(out_dir, 'slices.json')
    if os.path.exists(done) and not force:
        with open(manifest_path) as f:
            if json.load(f) == slices:
                print(f'{out_dir}: up to date, skipping')
                return
    if os.path.exists(done):
        os.remove(done)

    arrays = {name: np.lib.format.open_memmap(os.path.join(out_dir, name + '.npy'), mode='w+',
                                              dtype=np.float32, shape=(len(slices), N_PIX, N_PIX))
              for name in IMAGES}
    for i, entry in enumerate(tqdm(slices, desc=os.path.basename(out_dir))):
        sino = np.load(entry['sino']).astype(np.float32)
        arrays['label'][i] = full_fbp(sino)
        arrays['fbp_in'][i] = sparse_fbp(sino, cond)
        arrays['rls'][i] = op.rls(sino[cond] + 1.0, rls_beta, iters=rls_iters)
    for arr in arrays.values():
        arr.flush()
    del arrays

    with open(manifest_path, 'w') as f:
        json.dump(slices, f, indent=1)
    open(done, 'w').close()


def _stats(sd, offset, op_norm_sq, rls_beta, cond, n_sample=256):
    """Global intensity ranges, from the training patients only."""
    train = SplitData(sd, 'train')
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(train), size=min(n_sample, len(train)), replace=False))
    stats = {'cond_indices': cond, 'op_norm_sq': op_norm_sq, 'rls_beta': rls_beta}
    for name in IMAGES:
        vals = np.asarray(train.array(name)[idx])
        stats[f'{name}_lo'] = float(np.percentile(vals, 0.1))
        stats[f'{name}_hi'] = float(np.percentile(vals, 99.9))
        if name == 'label':
            # typical intensity of FBP(s + 1), the units the iterative methods work in
            stats['z_scale'] = float(np.percentile(vals + offset[None], 99.5))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--data_root', default='data', help='folder holding one subfolder per patient')
    parser.add_argument('--patients', default=None,
                        help='comma-separated patient folders in split order; skips discovery')
    parser.add_argument('--exclude', default=','.join(splits.DEFAULT_EXCLUDE),
                        help='folder names under --data_root that are not patients')
    parser.add_argument('--n_train', type=int, default=7)
    parser.add_argument('--n_val', type=int, default=1)
    parser.add_argument('--n_test', type=int, default=2)
    parser.add_argument('--test_stride', type=int, default=1, help='keep every Nth test slice')
    parser.add_argument('--rls_beta', type=float, default=1e-2,
                        help="smoothness weight of DOLCE's RLS condition, relative to ||A^T A||")
    parser.add_argument('--rls_iters', type=int, default=50)
    parser.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)),
                        help='processes computing sinograms')
    parser.add_argument('--force', action='store_true', help='rebuild splits that are up to date')
    args = parser.parse_args()

    sd = setting_dir(args)
    os.makedirs(sd, exist_ok=True)
    cond = cond_indices(args.angle_start, args.angle_end, args.angle_stride)
    patients = [p for p in args.patients.split(',') if p] if args.patients else None
    split = splits.make_split(args.data_root, args.n_train, args.n_val, args.n_test,
                              patients, tuple(args.exclude.split(',')))
    for name in SPLITS:
        print(f'{name:>5}: ' + ', '.join(os.path.basename(os.path.normpath(p)) for p in split[name]))
    print(f'measured rows ({len(cond)}): {cond}')

    # 1. Sinograms through SD-Flow's dicom_dataset, one process per series
    sino_cache = os.path.join(args.cache_root, 'sino')
    os.makedirs(sino_cache, exist_ok=True)
    jobs = [(series, sino_cache) for name in SPLITS for patient in split[name]
            for series in splits.series_dirs(patient)]
    with mp.get_context('spawn').Pool(max(1, min(args.workers, len(jobs)))) as pool:
        index_maps = dict(pool.map(_cache_series, jobs, chunksize=1))
    manifest = _manifest(split, index_maps, args.test_stride)

    with open(os.path.join(sd, 'split.json'), 'w') as f:
        json.dump({'data_root': args.data_root,
                   'cond_indices': cond,
                   'test_stride': args.test_stride,
                   'patients': {n: [os.path.basename(os.path.normpath(p)) for p in split[n]] for n in SPLITS},
                   'series': {n: sorted({os.path.join(s['patient'], s['series']) for s in manifest[n]})
                              for n in SPLITS},
                   'n_slices': {n: len(manifest[n]) for n in SPLITS}}, f, indent=2)

    # 2. Label, conditioning FBP and RLS images
    op = MeasuredOperator(cond)
    rls_beta = args.rls_beta * op.norm_sq()
    offset = fbp_of_ones()
    np.save(os.path.join(sd, 'K.npy'), offset)
    for name in SPLITS:
        _write_split(os.path.join(sd, name), manifest[name], cond, op, rls_beta, args.rls_iters, args.force)
    op_norm_sq = op.norm_sq()
    op.close()

    with open(os.path.join(sd, 'stats.json'), 'w') as f:
        json.dump(_stats(sd, offset, op_norm_sq, rls_beta, cond), f, indent=2)
    print(f'Done: {sd}  (' + ', '.join(f'{n} {len(manifest[n])} slices' for n in SPLITS) + ')')


if __name__ == '__main__':
    main()
