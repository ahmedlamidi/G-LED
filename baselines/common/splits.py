"""Patient discovery and the train / val / test split.

The data root holds one folder per patient in the LIDC-IDRI layout,
<patient>/<study>/<series>/*.dcm, e.g. data/LIDC-IDRI/LIDC-IDRI-0001/... In
natural sort order the first 10 patients are used: the first 7 train, the next
1 validation, the last 2 test. Only CT series are read: LIDC patients also hold
chest X-rays (DX / CR) and nodule segmentations (SEG), which are skipped.
"""
import os
import re

import SimpleITK as sitk

# Folders that are not patients of this study, should a root hold them
DEFAULT_EXCLUDE = ('LIDC-IDRI', 'Dataset', 'baselines_cache', 'sino_cache',
                   'sino_cache_test', 'ground_truth', 'test_data', 'extra_data')
# The label must be a full-dose image, so low-dose series are never used
LOW_DOSE = re.compile(r'low[\s_-]*dose|quarter', re.IGNORECASE)


def natural_key(name):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r'(\d+)', name)]


def _modality(path):
    """DICOM Modality (0008,0060) from the header only."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    try:
        reader.ReadImageInformation()
    except RuntimeError:
        return None
    if not reader.HasMetaDataKey('0008|0060'):
        return None
    return reader.GetMetaData('0008|0060').strip().upper()


def series_dirs(patient_dir):
    """Every folder under patient_dir that directly holds a CT series."""
    found = []
    for root, dirs, files in os.walk(patient_dir, followlinks=True):
        dirs.sort(key=natural_key)
        dcm = sorted(f for f in files if f.lower().endswith('.dcm'))
        if not dcm or LOW_DOSE.search(os.path.relpath(root, patient_dir)):
            continue
        if _modality(os.path.join(root, dcm[0])) == 'CT':
            found.append(root)
    return sorted(found, key=natural_key)


def discover_patients(data_root, exclude=DEFAULT_EXCLUDE, limit=None):
    """Patient folders holding at least one CT series, in natural order; stops at limit."""
    patients = []
    for name in sorted(os.listdir(data_root), key=natural_key):
        path = os.path.join(data_root, name)
        if not os.path.isdir(path) or name in exclude or name.startswith(('.', 'fbp_')):
            continue
        if series_dirs(path):
            patients.append(path)
            if limit is not None and len(patients) == limit:
                break
    return patients


def make_split(data_root, n_train=7, n_val=1, n_test=2, patients=None, exclude=DEFAULT_EXCLUDE):
    """With no explicit list, the first n_train + n_val + n_test patients under data_root."""
    needed = n_train + n_val + n_test
    explicit = patients is not None
    if not explicit:
        patients = discover_patients(data_root, exclude, limit=needed)
    if len(patients) < needed or (explicit and len(patients) != needed):
        listing = '\n  '.join(patients) if patients else '(none)'
        raise SystemExit(
            f'Need {needed} patient folders with a CT series under {data_root}, got {len(patients)}:\n'
            f'  {listing}\nPoint --data_root at the folder holding the patients, or list them with --patients.')
    return {'train': patients[:n_train],
            'val': patients[n_train:n_train + n_val],
            'test': patients[n_train + n_val:needed]}
