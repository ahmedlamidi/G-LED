"""Patient discovery and the train / val / test split.

Patients are the immediate subfolders of the data root that hold DICOM files
somewhere below them. In natural sort order (patient_2 before patient_10), the
first 7 are train, the next 1 validation and the last 2 test.
"""
import os
import re

# Folders under data/ that are not patients of this study
DEFAULT_EXCLUDE = ('LIDC-IDRI', 'Dataset', 'baselines_cache', 'sino_cache',
                   'sino_cache_test', 'ground_truth', 'test_data', 'extra_data')
# The label must be a full-dose image, so low-dose series are never used
LOW_DOSE = re.compile(r'low[\s_-]*dose|quarter', re.IGNORECASE)


def natural_key(name):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r'(\d+)', name)]


def series_dirs(patient_dir):
    """Every folder under patient_dir that directly holds .dcm files."""
    found = []
    for root, dirs, files in os.walk(patient_dir, followlinks=True):
        dirs.sort(key=natural_key)
        if not any(f.lower().endswith('.dcm') for f in files):
            continue
        if LOW_DOSE.search(os.path.relpath(root, patient_dir)):
            continue
        found.append(root)
    return sorted(found, key=natural_key)


def discover_patients(data_root, exclude=DEFAULT_EXCLUDE):
    patients = []
    for name in sorted(os.listdir(data_root), key=natural_key):
        path = os.path.join(data_root, name)
        if not os.path.isdir(path) or name in exclude or name.startswith(('.', 'fbp_')):
            continue
        if series_dirs(path):
            patients.append(path)
    return patients


def make_split(data_root, n_train=7, n_val=1, n_test=2, patients=None, exclude=DEFAULT_EXCLUDE):
    if patients is None:
        patients = discover_patients(data_root, exclude)
    expected = n_train + n_val + n_test
    if len(patients) != expected:
        listing = '\n  '.join(patients) if patients else '(none)'
        raise SystemExit(
            f'Expected {expected} patient folders under {data_root}, found {len(patients)}:\n  {listing}\n'
            'Point --data_root at the folder holding the patients, or list them with --patients.')
    return {'train': patients[:n_train],
            'val': patients[n_train:n_train + n_val],
            'test': patients[n_train + n_val:]}
