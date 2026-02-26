import os
from pathlib import Path
import numpy as np
import astra
import datetime
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, CTImageStorage, ExplicitVRLittleEndian
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def convert_hu_to_mu(ct_slice):
    """Convert HU to linear attenuation coefficient (mu).
    mu_water ~ 0.02 mm^-1 at typical CT energies.
    """
    ct_slice = ct_slice.astype(np.float32, copy=False)
    mu = 0.02 * (1.0 + ct_slice / 1000.0)
    mu = np.clip(mu, 0, None)
    return mu


def create_sinogram_from_dicom(ct_slice, dx, dy):
    """Convert a CT slice to sinogram (matching saved_ground_truth.py)."""
    mu_slice = convert_hu_to_mu(ct_slice)
    
    H, W = list(mu_slice.shape)[:2]
    
    DSO = 1000
    ODD = 600  
    angles_deg = np.arange(0, 360, 360/512, dtype=np.float32)
    angles = np.deg2rad(angles_deg)
    
    vol_geom = astra.create_vol_geom(H, W,
        -W * dx / 2.0, W * dx / 2.0,
        -H * dy / 2.0, H * dy / 2.0
    )
    
    # Detector should cover the full object diagonal (matching saved_ground_truth.py)
    det_count = 512
    det_spacing = dx  
    
    proj_geom = astra.create_proj_geom('fanflat', det_spacing, det_count, angles, DSO, ODD)
    projector_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

    slice2d = np.ascontiguousarray(mu_slice, dtype=np.float32)
    sid = astra.data2d.create('-vol', vol_geom, slice2d)
    sino_id, sino = astra.create_sino(sid, projector_id) 
    
    astra.data2d.delete(sino_id)
    astra.data2d.delete(sid)
    astra.projector.delete(projector_id)
    
    return sino


def load_series_from(path):
    """Load DICOM series from path."""
    if path.endswith(".dcm"):
        img = sitk.ReadImage(path)
    else:
        series_dir = Path(path)
        sitk_reader = sitk.ImageSeriesReader()
        file_names = sitk_reader.GetGDCMSeriesFileNames(str(series_dir))
        sitk_reader.SetFileNames(file_names)
        img = sitk_reader.Execute()
    
    vol_zyx = sitk.GetArrayFromImage(img)  
    spacing_x, spacing_y, spacing_z = img.GetSpacing()
    return vol_zyx, (spacing_x, spacing_y, spacing_z)


def get_ground_truth_sinograms(data_path="data/Dataset", num_slices=20):
    """Load first num_slices from Dataset and create sinograms with their min/max values."""
    vol_zyx, spacing = load_series_from(data_path)
    
    sinograms = []
    sino_ranges = []
    ct_dims = []  # Store original CT dimensions
    
    for ind in range(min(num_slices, len(vol_zyx))):
        ct_slice = vol_zyx[ind]
        H, W = ct_slice.shape[:2]
        ct_dims.append((H, W))
        
        sino = create_sinogram_from_dicom(ct_slice, spacing[0], spacing[1])
        sino_min, sino_max = sino.min(), sino.max()
        sinograms.append(sino)
        sino_ranges.append((sino_min, sino_max))
        print(f"  Slice {ind}: CT shape ({H}, {W}), sino shape {sino.shape}, range [{sino_min:.6f}, {sino_max:.6f}]")
    
    return sinograms, sino_ranges, spacing, ct_dims


def save_ct_volume_as_dicom(
    volume_zyx: np.ndarray,
    spacing_dzyx: tuple,
    out_dir: Path,
    filename: str,
    *,
    patient_id: str = "Generated",
    study_uid: str = None,
    series_uid: str = None,
    series_description: str = "Generated Reconstruction",
    manufacturer: str = "Diffusion Model Pipeline",
):
    """
    Save a 2D CT slice (HU) as a DICOM file.
    """
    if study_uid is None:
        study_uid = generate_uid()
    if series_uid is None:
        series_uid = generate_uid()

    out_dir.mkdir(parents=True, exist_ok=True)

    Y, X = volume_zyx.shape
    Z = 1
    dz, dy, dx = spacing_dzyx

    now = datetime.datetime.now()
    study_date = now.strftime("%Y%m%d")
    study_time = now.strftime("%H%M%S")
    
    frame_of_reference_uid = generate_uid()

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

    ds.PatientName = patient_id
    ds.PatientID = patient_id
    ds.PatientBirthDate = ""
    ds.PatientSex = ""

    ds.StudyInstanceUID = study_uid
    ds.StudyDate = study_date
    ds.StudyTime = study_time
    ds.ReferringPhysicianName = ""
    ds.StudyID = "1"
    ds.AccessionNumber = ""

    ds.SeriesInstanceUID = series_uid
    ds.SeriesDate = study_date
    ds.SeriesTime = study_time
    ds.Modality = "CT"
    ds.SeriesDescription = series_description
    ds.SeriesNumber = 1
    ds.Manufacturer = manufacturer

    ds.FrameOfReferenceUID = frame_of_reference_uid
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
    ds.WindowCenter = "-600"
    ds.WindowWidth = "1600"

    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    slice_data = volume_zyx.astype(np.int16)
    ds.PixelData = slice_data.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(filepath), write_like_original=False)
    
    return filepath


def projection(img_array, spacing, ct_dims=None):
    """Perform FBP reconstruction from sinogram (matching saved_ground_truth.py)."""
    from typing import Any, Dict
    
    sino_AD = np.asarray(img_array, dtype=np.float32)

    # Sinogram dimensions: (num_angles, num_detectors)
    num_angles = sino_AD.shape[0]
    num_detectors = sino_AD.shape[1]

    DSO = 1000  
    ODD = 600  

    # Use same angle generation as saved_ground_truth.py (always 512 angles for 360 degrees)
    angles_deg = np.arange(0, 360, (360/512), dtype=np.float32)
    angles = np.deg2rad(angles_deg)
    
    print(f"  FBP: {num_angles} angles, {num_detectors} detectors")

    spacing_xyz = spacing

    # Use original CT dimensions if provided, else default to 512x512
    if ct_dims is not None:
        H, W = ct_dims
    else:
        H = 512
        W = 512

    dx = spacing_xyz[0]
    dy = spacing_xyz[1]

    # Create volume geometry matching original CT slice dimensions
    vol_geom = astra.create_vol_geom(H, W,
        -W * dx / 2.0,  W * dx / 2.0,   # x_min, x_max
        -H * dy / 2.0,  H * dy / 2.0    # y_min, y_max
    )
        
    # Detector count from sinogram (not hardcoded)
    det_count = num_detectors
    det_spacing = dx  

    proj_geom = astra.create_proj_geom('fanflat', det_spacing, det_count, angles, DSO, ODD)

    projector_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

    sinogram_id = astra.data2d.create('-sino', proj_geom, sino_AD)
    recon_id = astra.data2d.create('-vol', vol_geom)

    cfg: Dict[str, Any] = astra.astra_dict("FBP_CUDA")
    cfg["ProjectionDataId"] = sinogram_id
    cfg["ReconstructionDataId"] = recon_id
    options: Dict[str, Any] = {}
    options["FilterType"] = "Shepp-Logan"
    cfg["option"] = options
    
    alg_id = astra.algorithm.create(cfg)
    try:
        astra.algorithm.run(alg_id)
        result = astra.data2d.get(recon_id).astype(np.float32)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(recon_id)
        astra.projector.delete(projector_id)

    return result


def convert_mu_to_hu_recon(mu, mu_water=0.02):
    """Convert linear attenuation coefficient to Hounsfield Units."""
    mu = mu.astype(np.float32, copy=False)

    if mu_water is None:
        center = mu[
            mu.shape[0]//4 : 3*mu.shape[0]//4,
            mu.shape[1]//4 : 3*mu.shape[1]//4
        ]
        mu_water = np.median(center)

    mu = np.clip(mu, 0, None)
    hu = 1000.0 * (mu / mu_water - 1.0)
    hu = np.clip(hu, -1024, 3071)

    return hu


def denormalize_sinogram(sino_normalized, sino_min, sino_max):
    """Convert sinogram from [-1, 1] range back to original scale.
    
    The original normalization was:
        sino = 2.0 * (sino - sino_min) / (sino_max - sino_min) - 1.0
    
    So to denormalize:
        sino = (sino_normalized + 1.0) / 2.0 * (sino_max - sino_min) + sino_min
    """
    # Convert from [-1, 1] to [0, 1]
    sino = (sino_normalized + 1.0) / 2.0
    
    # Scale to original range
    sino = sino * (sino_max - sino_min) + sino_min
    
    return sino


def process_batch(batch_folder, output_dir, sino_min, sino_max, spacing=(1.0, 1.0, 1.0), ct_dims=None):
    """Process a single batch folder and save as DICOM."""
    
    # Load generated reconstruction and condition
    gen_path = os.path.join(batch_folder, 'recon_micro_0.npy')
    cond_path = os.path.join(batch_folder, 'recon_micro_0gt.npy')
    
    img_generated = np.load(gen_path)
    img_cond = np.load(cond_path)
    
    # Squeeze to 2D
    if img_generated.ndim == 5:
        img_generated = img_generated[0, 0, 0]
    elif img_generated.ndim == 4:
        img_generated = img_generated[0, 0]
    elif img_generated.ndim == 3:
        img_generated = img_generated[0]
    
    if img_cond.ndim == 5:
        img_cond = img_cond[0, 0, 0]
    elif img_cond.ndim == 4:
        img_cond = img_cond[0, 0]
    elif img_cond.ndim == 3:
        img_cond = img_cond[0]
    
    # Concatenate condition and generated along axis 1 (detectors)
    # Condition = first half of detectors, Generated = second half of detectors
    combined = np.concatenate([img_cond, img_generated], axis=1)
    
    print(f"  Sinogram shape: {combined.shape}")
    print(f"  Value range: [{combined.min():.4f}, {combined.max():.4f}]")
    
    # Denormalize from [-1, 1] to original sinogram values using actual min/max
    combined_denorm = denormalize_sinogram(combined, sino_min, sino_max)
    
    print(f"  Denorm range: [{combined_denorm.min():.6f}, {combined_denorm.max():.6f}]")
    
    # Perform FBP reconstruction on the full sinogram
    recon = projection(combined_denorm, spacing, ct_dims)
    
    # Convert mu to HU
    recon_hu = convert_mu_to_hu_recon(recon)
    recon_hu = np.nan_to_num(recon_hu, nan=-1024, posinf=3071, neginf=-1024)
    
    return recon_hu, combined


if __name__ == '__main__':
    # Base folder containing all batch folders (generated outputs)
    base_folder = 'output/feb_19_720_820_model/diffusion_folder/experiment_final/contour'
    
    # Output folder for all DICOM files (in experiment_final)
    dicom_output_folder = Path('output/feb_19_720_820_model/diffusion_folder/experiment_final/dicom_output')
    dicom_output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth sinograms from Dataset to get actual min/max values and CT dimensions
    print("Loading ground truth sinograms from Dataset...")
    gt_sinograms, sino_ranges, spacing, ct_dims = get_ground_truth_sinograms("data/test_data", num_slices=20)
    print(f"Loaded {len(gt_sinograms)} ground truth sinograms")
    
    # Get all batch folders and sort them
    batch_folders = [f for f in os.listdir(base_folder) if f.startswith('batch') and os.path.isdir(os.path.join(base_folder, f))]
    batch_folders.sort(key=lambda x: int(x.replace('batch', '')))
    
    print(f"Found {len(batch_folders)} batch folders")
    
    # Generate shared UIDs for the study
    study_uid = generate_uid()
    
    # List to store metrics for all batches
    metrics_list = []
    
    for batch_name in batch_folders:
        batch_path = os.path.join(base_folder, batch_name)
        batch_num = int(batch_name.replace('batch', ''))
        
        # Make sure we have the corresponding ground truth
        if batch_num >= len(sino_ranges):
            print(f"Skipping {batch_name}: no ground truth available")
            continue
        
        sino_min, sino_max = sino_ranges[batch_num]
        ct_dim = ct_dims[batch_num]
        
        try:
            print(f"Processing {batch_name} (CT dims {ct_dim}, sino range [{sino_min:.6f}, {sino_max:.6f}])...")
            
            # Process the generated batch with correct normalization and CT dimensions
            recon_hu, combined_sino = process_batch(batch_path, dicom_output_folder, sino_min, sino_max, spacing, ct_dim)
            
            # Save generated as DICOM
            series_uid = generate_uid()
            dicom_filename = f"{batch_name}_generated.dcm"
            
            filepath = save_ct_volume_as_dicom(
                recon_hu,
                spacing,
                dicom_output_folder,
                dicom_filename,
                patient_id=f"Generated_{batch_name}",
                study_uid=study_uid,
                series_uid=series_uid,
                series_description=f"Generated Reconstruction - {batch_name}"
            )
            
            print(f"  Saved generated DICOM to: {filepath}")
            print(f"  HU range: [{recon_hu.min():.1f}, {recon_hu.max():.1f}]")
            
            # Also reconstruct and save ground truth as DICOM
            gt_sino = gt_sinograms[batch_num]
            gt_recon = projection(gt_sino, spacing, ct_dim)
            gt_hu = convert_mu_to_hu_recon(gt_recon)
            gt_hu = np.nan_to_num(gt_hu, nan=-1024, posinf=3071, neginf=-1024)
            
            # Save ground truth as DICOM
            gt_series_uid = generate_uid()
            gt_dicom_filename = f"{batch_name}_ground_truth.dcm"
            
            gt_filepath = save_ct_volume_as_dicom(
                gt_hu,
                spacing,
                dicom_output_folder,
                gt_dicom_filename,
                patient_id=f"GroundTruth_{batch_name}",
                study_uid=study_uid,
                series_uid=gt_series_uid,
                series_description=f"Ground Truth - {batch_name}"
            )
            
            print(f"  Saved ground truth DICOM to: {gt_filepath}")
            print(f"  GT HU range: [{gt_hu.min():.1f}, {gt_hu.max():.1f}]")
            
            # Calculate SSIM and PSNR between generated and ground truth
            # Normalize both to [0, 1] for metric calculation
            def normalize(img):
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    return (img - img_min) / (img_max - img_min)
                return img
            
            recon_norm = normalize(recon_hu)
            gt_norm = normalize(gt_hu)
            
            ssim_value = ssim(gt_norm, recon_norm, data_range=1.0)
            psnr_value = psnr(gt_norm, recon_norm, data_range=1.0)
            
            metrics_list.append({
                'batch': batch_name,
                'ssim': ssim_value,
                'psnr': psnr_value
            })
            
            print(f"  SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f} dB")
            
        except FileNotFoundError as e:
            print(f"Error processing {batch_name}: File not found - {e}")
        except Exception as e:
            print(f"Error processing {batch_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save metrics to file
    metrics_file = dicom_output_folder / 'metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("Batch\tSSIM\tPSNR (dB)\n")
        f.write("-" * 40 + "\n")
        for m in metrics_list:
            f.write(f"{m['batch']}\t{m['ssim']:.4f}\t{m['psnr']:.2f}\n")
        
        # Calculate and write averages
        if metrics_list:
            avg_ssim = np.mean([m['ssim'] for m in metrics_list])
            avg_psnr = np.mean([m['psnr'] for m in metrics_list])
            f.write("-" * 40 + "\n")
            f.write(f"Average\t{avg_ssim:.4f}\t{avg_psnr:.2f}\n")
    
    print(f"\nMetrics saved to {metrics_file}")
    
    # Also print summary
    if metrics_list:
        avg_ssim = np.mean([m['ssim'] for m in metrics_list])
        avg_psnr = np.mean([m['psnr'] for m in metrics_list])
        print(f"\n=== Summary ===")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    print(f"\nDone! All DICOM files saved to {dicom_output_folder}")
