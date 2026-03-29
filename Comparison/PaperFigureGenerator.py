import sys
import os
import csv
import datetime
import numpy as np
import astra
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, CTImageStorage, ExplicitVRLittleEndian

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from dicom_preprocess import load_series_from, convert_hu_to_mu


def convert_mu_to_hu(mu):
    """Inverse of convert_hu_to_mu: HU = (mu / 0.02 - 1) * 1000."""
    return (mu / 0.02 - 1.0) * 1000.0


def save_ct_as_dicom(
    slice_hu: np.ndarray,
    spacing_xyz: tuple,
    out_dir: Path,
    filename: str,
    series_description: str = "FBP Reconstruction",
):
    """Save a 2D CT slice (HU) as a DICOM file."""
    out_dir.mkdir(parents=True, exist_ok=True)

    Y, X = slice_hu.shape
    dx, dy, dz = spacing_xyz

    now = datetime.datetime.now()
    study_date = now.strftime("%Y%m%d")
    study_time = now.strftime("%H%M%S")

    filepath = out_dir / filename

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.ImplementationVersionName = "PYDICOM"

    ds = FileDataset(str(filepath), {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.ImageType = ["DERIVED", "SECONDARY", "AXIAL"]
    ds.PatientName = "PaperFigure"
    ds.PatientID = "PaperFigure"
    ds.PatientBirthDate = ""
    ds.PatientSex = ""

    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = study_date
    ds.StudyTime = study_time
    ds.ReferringPhysicianName = ""
    ds.StudyID = "1"
    ds.AccessionNumber = ""

    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDate = study_date
    ds.SeriesTime = study_time
    ds.Modality = "CT"
    ds.SeriesDescription = series_description
    ds.SeriesNumber = 1
    ds.Manufacturer = "FBP Pipeline"

    ds.FrameOfReferenceUID = generate_uid()
    ds.PositionReferenceIndicator = ""
    ds.InstitutionName = ""
    ds.StationName = ""
    ds.ManufacturerModelName = ""

    ds.InstanceNumber = 1
    ds.PatientOrientation = ""
    ds.ContentDate = study_date
    ds.ContentTime = study_time
    ds.AcquisitionDate = study_date
    ds.AcquisitionTime = study_time
    ds.AcquisitionNumber = 1

    ds.PixelSpacing = [float(dy), float(dx)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.SliceThickness = float(dz)
    ds.SliceLocation = 0.0
    ds.PatientPosition = "HFS"

    ds.Rows = Y
    ds.Columns = X
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed

    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.RescaleType = "HU"
    ds.KVP = ""
    ds.WindowCenter = "40"
    ds.WindowWidth = "400"

    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    slice_data = np.clip(slice_hu, -1024, 3071).astype(np.int16)
    ds.PixelData = slice_data.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(filepath), write_like_original=False)

    return filepath


# ── ASTRA helpers (from FBP.py) ──────────────────────────────────────────────

def build_geometry(H, W, dx, dy, detector_count, angle_step):
    DSO = 1000
    ODD = 600
    angles_deg = np.arange(0, 360, angle_step, dtype=np.float32)
    angles = np.deg2rad(angles_deg)

    vol_geom = astra.create_vol_geom(
        H, W,
        -W * dx / 2.0, W * dx / 2.0,
        -H * dy / 2.0, H * dy / 2.0,
    )
    proj_geom = astra.create_proj_geom(
        'fanflat', dx, detector_count, angles, DSO, ODD
    )
    return vol_geom, proj_geom, angles


def forward_project(mu_slice, vol_geom, proj_geom):
    slice2d = np.ascontiguousarray(mu_slice, dtype=np.float32)
    sid = astra.data2d.create('-vol', vol_geom, slice2d)
    proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
    sino_id, sino = astra.create_sino(sid, proj_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(sid)
    astra.projector.delete(proj_id)
    return sino


def fbp_reconstruct(sinogram, vol_geom, proj_geom):
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    cfg['option'] = {'ShortScan': False}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recon = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)
    return recon


def build_sparse_geometry(full_angles, cond_indices, dx, detector_count):
    DSO = 1000
    ODD = 600
    sparse_angles = full_angles[cond_indices]
    proj_geom = astra.create_proj_geom(
        'fanflat', dx, detector_count, sparse_angles, DSO, ODD
    )
    return proj_geom


# ── Core logic ────────────────────────────────────────────────────────────────

def process_slice(ct_slice_hu, mu_slice, vol_geom, proj_geom_full, full_angles,
                  cond_indices, dx, detector_count):
    """Run FBP for full and degraded views, return metrics and HU reconstructions."""
    sino_full = forward_project(mu_slice, vol_geom, proj_geom_full)
    recon_full_mu = fbp_reconstruct(sino_full, vol_geom, proj_geom_full)

    sino_sparse = sino_full[cond_indices, :]
    proj_geom_sparse = build_sparse_geometry(
        full_angles, cond_indices, dx, detector_count
    )
    recon_sparse_mu = fbp_reconstruct(sino_sparse, vol_geom, proj_geom_sparse)

    recon_full_hu = convert_mu_to_hu(recon_full_mu)
    recon_sparse_hu = convert_mu_to_hu(recon_sparse_mu)

    hu_data_range = recon_full_hu.max() - recon_full_hu.min()
    hu_ssim = ssim(recon_full_hu, recon_sparse_hu, data_range=hu_data_range)
    hu_psnr = psnr(recon_full_hu, recon_sparse_hu, data_range=hu_data_range)

    recon_data_range = recon_full_mu.max() - recon_full_mu.min()
    recon_ssim = ssim(recon_full_mu, recon_sparse_mu, data_range=recon_data_range)
    recon_psnr = psnr(recon_full_mu, recon_sparse_mu, data_range=recon_data_range)

    metrics = {
        'recon_ssim': recon_ssim, 'recon_psnr': recon_psnr,
        'hu_ssim': hu_ssim, 'hu_psnr': hu_psnr,
    }
    return metrics, recon_full_hu, recon_sparse_hu


def run_setting(detector_count, angle_step, total_series, dicom_dir,
                cutoff_pct=None, step=None):
    """
    cutoff_pct only (step=None) : limited-view  (contiguous first N% of angles)
    step only (cutoff_pct=None) : sparse-view   (every Nth angle across full 360°)
    """
    total_angles = 720

    if step is None:
        # Limited view
        cutoff = int(total_angles * cutoff_pct / 100)
        cond_indices = list(range(cutoff))
        degrees = cutoff_pct / 100.0 * 360
        label = f"limited_{int(degrees)}deg"
        desc = f"Limited view: {int(degrees)}° = {len(cond_indices)} of {total_angles} angles"
    else:
        # Sparse view
        cond_indices = list(range(0, total_angles, step))
        label = f"sparse_step{step}"
        desc = f"Sparse view: every {step}th angle = {len(cond_indices)} of {total_angles} angles"

    n_known = len(cond_indices)
    print(f"\n{'='*80}")
    print(desc)
    print(f"{'='*80}")

    # Use only the first slice from the first series
    vol_zyx, spacing = total_series[0]
    dx, dy, dz = spacing
    H, W = vol_zyx[0].shape[:2]

    vol_geom, proj_geom_full, full_angles = build_geometry(
        H, W, dx, dy, detector_count, angle_step
    )

    ct_slice_hu = vol_zyx[0].astype(np.float32)
    mu_slice = convert_hu_to_mu(ct_slice_hu)

    metrics, recon_full_hu, recon_sparse_hu = process_slice(
        ct_slice_hu, mu_slice, vol_geom, proj_geom_full, full_angles,
        cond_indices, dx, detector_count
    )

    print(f"  Slice 0: "
          f"Recon SSIM={metrics['recon_ssim']:.4f} PSNR={metrics['recon_psnr']:.2f} | "
          f"HU SSIM={metrics['hu_ssim']:.4f} PSNR={metrics['hu_psnr']:.2f}")

    # Save degraded reconstruction as DICOM
    dcm_name = f"{label}_slice0.dcm"
    save_ct_as_dicom(
        recon_sparse_hu, spacing, dicom_dir, dcm_name,
        series_description=f"FBP {desc}",
    )
    print(f"  -> Saved DICOM: {dcm_name}")

    # Save full reference once
    ref_name = "full_reference_slice0.dcm"
    ref_path = dicom_dir / ref_name
    if not ref_path.exists():
        save_ct_as_dicom(
            recon_full_hu, spacing, dicom_dir, ref_name,
            series_description="FBP Full 720 angles (reference)",
        )
        print(f"  -> Saved reference DICOM: {ref_name}")

    return label, n_known, metrics


def main():
    data_path = "data/test_data"
    detector_count = 816
    angle_step = 360 / 720

    dicom_dir = Path("Comparison/PaperFigures_DICOM")

    total_series = load_series_from(data_path)

    # ── Settings ──────────────────────────────────────────────────────────────
    # Sparse view: every Nth angle across full 360°
    sparse_steps = [2, 4, 8, 10]

    # Limited view: contiguous angles (percentage of 360°)
    #   180° = 50%, 90° = 25%, 45° = 12.5%
    limited_pcts = [50, 25, 12.5]
    # ──────────────────────────────────────────────────────────────────────────

    results = []

    for step in sparse_steps:
        label, n, m = run_setting(
            detector_count, angle_step, total_series, dicom_dir,
            cutoff_pct=None, step=step
        )
        results.append((label, n, m))

    for pct in limited_pcts:
        label, n, m = run_setting(
            detector_count, angle_step, total_series, dicom_dir,
            cutoff_pct=pct, step=None
        )
        results.append((label, n, m))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FINAL SUMMARY (slice 0)")
    print(f"{'='*80}")
    print(f"{'Setting':<25} {'Angles':<8} {'Recon SSIM':<12} {'Recon PSNR':<12} {'HU SSIM':<12} {'HU PSNR':<10}")
    print("-" * 80)
    for label, n, m in results:
        print(f"{label:<25} {n:<8} {m['recon_ssim']:<12.4f} {m['recon_psnr']:<12.2f} "
              f"{m['hu_ssim']:<12.4f} {m['hu_psnr']:<10.2f}")

    # Save summary CSV
    csv_path = dicom_dir / "summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['setting', 'num_angles', 'recon_ssim', 'recon_psnr', 'hu_ssim', 'hu_psnr'])
        for label, n, m in results:
            writer.writerow([label, n, m['recon_ssim'], m['recon_psnr'],
                             m['hu_ssim'], m['hu_psnr']])

    print(f"\nDICOM files and summary saved to: {dicom_dir}")


if __name__ == '__main__':
    main()
