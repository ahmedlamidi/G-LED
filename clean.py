from pathlib import Path
import SimpleITK as sitk
import os
# Standalone script: scan all .dcm files in a directory tree and delete unreadable ones
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Delete unreadable DICOM files in a directory tree.")
    parser.add_argument('--root', type=str, default="data/LIDC-IDRI", help="Root directory to scan for DICOM files.")
    args = parser.parse_args()

    root_dir = args.root
    print(f"Scanning for DICOM files in: {root_dir}")
    deleted = 0
    checked = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.dcm'):
                fpath = os.path.join(dirpath, fname)
                checked += 1
                try:
                    sitk.ReadImage(fpath)
                except Exception as e:
                    print(f"Unreadable DICOM: {fpath}\n{e}\nDeleting...")
                    try:
                        os.remove(fpath)
                        deleted += 1
                    except Exception as de:
                        print(f"Failed to delete {fpath}: {de}")
    print(f"Checked {checked} DICOM files. Deleted {deleted} unreadable files.")
