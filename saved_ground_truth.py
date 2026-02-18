import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import imageio.v3 as iio
import astra
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pdb
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, cast
import datetime
import numpy as np
import pydicom
from pathlib import Path
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, CTImageStorage, ExplicitVRLittleEndian
from pydicom.uid import generate_uid
import datetime
import numpy as np
import pydicom
from pathlib import Path
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, CTImageStorage, ExplicitVRLittleEndian

def save_ct_volume_as_dicom(
    volume_zyx: np.ndarray,
    spacing_dzyx: tuple[float, float, float],
    out_dir: Path,
    *,
    patient_id: str,
    study_uid: str,
    series_uid: str,
    series_description: str = "FBP Reconstruction",
    manufacturer: str = "Custom FBP Pipeline",
):
    """
    Save a 3D CT volume (HU) as a DICOM series.

    - volume_zyx: CT volume as [Z, Y, X], HU values (# signed int16)
    - spacing_dzyx: (dz, dy, dx) in mm
    - out_dir: where to save .dcm files
    - patient_id/study_uid/series_uid: provided externally
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    Y, X = volume_zyx.shape
    Z = 1
    dz, dy, dx = spacing_dzyx

    # For consistent timing
    now = datetime.datetime.now()
    study_date = now.strftime("%Y%m%d")
    study_time = now.strftime("%H%M%S")
    
    # Generate shared FrameOfReferenceUID for the series
    frame_of_reference_uid = generate_uid()

    for z in range(Z):
        # Build unique file path
        filename = out_dir / f"slice_1024_512.dcm"

        # Create minimal file meta information
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.ImplementationVersionName = "PYDICOM"

        # Build the dataset
        ds = FileDataset(str(filename), {}, file_meta=file_meta, preamble=b"\0" * 128)

        # --- Required tags for viewer compatibility ---
        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.ImageType = ["DERIVED", "SECONDARY", "AXIAL"]

        # --- Patient Module (Required) ---
        ds.PatientName = patient_id
        ds.PatientID = patient_id
        ds.PatientBirthDate = ""
        ds.PatientSex = ""

        # --- General Study Module (Required) ---
        ds.StudyInstanceUID = study_uid
        ds.StudyDate = study_date
        ds.StudyTime = study_time
        ds.ReferringPhysicianName = ""
        ds.StudyID = "1"
        ds.AccessionNumber = ""

        # --- General Series Module (Required) ---
        ds.SeriesInstanceUID = series_uid
        ds.SeriesDate = study_date
        ds.SeriesTime = study_time
        ds.Modality = "CT"
        ds.SeriesDescription = series_description
        ds.SeriesNumber = 1
        ds.Manufacturer = manufacturer

        # --- Frame of Reference Module (Required for CT) ---
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.PositionReferenceIndicator = ""

        # --- General Equipment Module ---
        ds.InstitutionName = ""
        ds.StationName = ""
        ds.ManufacturerModelName = ""

        # --- General Image Module ---
        ds.InstanceNumber = z + 1
        ds.PatientOrientation = ""
        ds.ContentDate = study_date
        ds.ContentTime = study_time
        ds.AcquisitionDate = study_date
        ds.AcquisitionTime = study_time
        ds.AcquisitionNumber = 1

        # --- Image Plane Module (Required for CT) ---
        ds.PixelSpacing = [float(dy), float(dx)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [0.0, 0.0, float(z * dz)]
        ds.SliceThickness = float(dz)
        ds.SliceLocation = float(z * dz)
        ds.PatientPosition = "HFS"

        # --- Image Pixel Module (Required) ---
        ds.Rows = Y
        ds.Columns = X
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # signed

        # --- CT Image Module ---
        ds.RescaleIntercept = "0"
        ds.RescaleSlope = "1"
        ds.RescaleType = "HU"
        ds.KVP = ""
        ds.WindowCenter = "-600"
        ds.WindowWidth = "1600"

        # --- SOP Common Module (Required) ---
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

        # Convert to signed 16-bit
        slice_data = volume_zyx.astype(np.int16)
        print(f"Slice {z} shape: {slice_data.shape}, Byte length: {len(slice_data.tobytes())}")
        ds.PixelData = slice_data.tobytes()

        # Write the file
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(str(filename), write_like_original=False)



def projection(img_array, spacing):
    sino_AD = np.asarray(img_array, dtype=np.float32)

    # Sinogram dimensions: (num_angles, num_detectors)
    num_angles = sino_AD.shape[0]
    num_detectors = sino_AD.shape[1]

    DSO = 1000  
    ODD = 600  

    angles_deg = np.arange(0, 360, (360/1024), dtype=np.float32)
    angles = np.deg2rad(angles_deg)  # ASTRA expects radians

    spacing_xyz = spacing

    # Reconstruction volume size (typically same as detector count for square output)
    H =  512
    W = 512

    dx = spacing_xyz[0]
    dy = spacing_xyz[1]

    # generate params for the second part
    vol_geom = astra.create_vol_geom(H, W,
        -W * dx / 2.0,  W * dx / 2.0,   # x_min, x_max
        -H * dy / 2.0,  H * dy / 2.0    # y_min, y_max
    )
        
    # Detector count must match sinogram width
    det_count = num_detectors
    det_spacing = dx  

    #proj_geom = astra.create_proj_geom('parallel', det_spacing, det_count, angles)
    proj_geom = astra.create_proj_geom('fanflat', det_spacing, det_count, angles, DSO, ODD)

    projector_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

    sinogram_id = astra.data2d.create('-sino', proj_geom, sino_AD)
    recon_id = astra.data2d.create('-vol', vol_geom)

    cfg: Dict[str, Any] = astra.astra_dict("FBP_CUDA")
    cfg["ProjectionDataId"] = sinogram_id
    cfg["ReconstructionDataId"] = recon_id
    #cfg["ProjectorId"] = projector_id
    options: Dict[str, Any] = {}
    options["FilterType"] = "Shepp-Logan"
    cfg["option"] = options
    alg_id = astra.algorithm.create(cfg)
    try:
        astra.algorithm.run(alg_id)
        result = astra.data2d.get(recon_id).astype(np.float32)
    except:
        print("Not done")
    finally:
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(recon_id)

    arr = result
    return arr



def read_dicom_as_numpy(path):
    if path.endswith(".dcm"):
        img = sitk.ReadImage(path)
    else:
        series_dir = Path(path)
        sitk_reader = sitk.ImageSeriesReader()
        file_names = sitk_reader.GetGDCMSeriesFileNames(str(series_dir))
        sitk_reader.SetFileNames(file_names)
        img = sitk_reader.Execute()
    vol_zyx = sitk.GetArrayFromImage(img)
    return vol_zyx

def reconstruct_from_projection(sino):
    # Assumes util/projection.py has a function 'backproject' that takes a sinogram and returns a reconstruction
    recon = projection(sino)
    exit(0)
    return recon

def convert_mu_to_hu(mu, mu_water=0.02):
    mu = mu.astype(np.float32, copy=False)

    # Auto-estimate mu_water if not provided
    if mu_water is None:
        # assume central region ≈ water
        center = mu[
            mu.shape[0]//4 : 3*mu.shape[0]//4,
            mu.shape[1]//4 : 3*mu.shape[1]//4
        ]
        mu_water = np.median(center)

    # Prevent division explosions
    mu = np.clip(mu, 0, None)

    hu = 1000.0 * (mu / mu_water - 1.0)
    hu = np.clip(hu, -1024, 3071)

    return hu

def convert_hu_to_mu(ct_slice):
    """Convert HU to linear attenuation coefficient (mu).
    mu_water ~ 0.02 mm^-1 at typical CT energies.
    """
    ct_slice = ct_slice.astype(np.float32, copy=False)
    # HU to mu: mu = mu_water * (1 + HU/1000)
    mu = 0.02 * (1.0 + ct_slice / 1000.0)
    mu = np.clip(mu, 0, None)  # mu cannot be negative
    return mu

#Need to convert to sinogram and visualize
def convert_sinogram(ct_slice, dx, dy, dz):
    
    #det count should be width of a pixel
    #dx_mm which is det_spacing is dx_mm
    
    # Convert HU to attenuation coefficients first
    mu_slice = convert_hu_to_mu(ct_slice)
   # mu_slice = ct_slice
    H, W = list(mu_slice.shape)[:2]
    
    DSO = 1000  
    ODD = 600  
    angles_deg = np.arange(0, 360, 360/1024, dtype=np.float32)
    angles = np.deg2rad(angles_deg)  # ASTRA expects radians
    
    # generate params for the second part
    vol_geom = astra.create_vol_geom( H, W,
        -W * dx / 2.0,  W* dx/ 2.0,   # x_min, x_max
        -H * dy / 2.0,  H * dy / 2.0    # y_min, y_max
    )
        
    # Detector should cover the full object diagonal
    det_count =  512
    det_spacing = dx  
    
    proj_geom = astra.create_proj_geom('fanflat', det_spacing, det_count, angles, DSO, ODD)
    
    projector_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

    slice2d = np.ascontiguousarray(mu_slice, dtype=np.float32)  # (H, W)
    sid = astra.data2d.create('-vol', vol_geom, slice2d)
    sino_id, sino = astra.create_sino(sid, projector_id) 
    
    # Cleanup ASTRA resources
    astra.data2d.delete(sino_id)
    astra.data2d.delete(sid)
    astra.projector.delete(projector_id)
    # print(sino.shape)
    return sino  # Return the sinogram

def load_series_from(path):
    #set path of the Dicom series
    if path.endswith(".dcm"):
        img = sitk.ReadImage(path)
    else:
        series_dir = Path(path)
        sitk_reader = sitk.ImageSeriesReader()
        file_names = sitk_reader.GetGDCMSeriesFileNames(str(series_dir))
        sitk_reader.SetFileNames(file_names)
        img = sitk_reader.Execute()
    # file_names = sitk_reader.GetGDCMSeriesFileNames(str(series_dir))
    
    vol_zyx = sitk.GetArrayFromImage(img)  
    spacing_x, spacing_y, spacing_z = img.GetSpacing()
    #visualize(vol_zyx[0])
    return vol_zyx, (spacing_x, spacing_y, spacing_z)



def main(dicom_path):
    vol_zyx, spacing = load_series_from(dicom_path)
        
        # Convert all slices to sinograms
    for ind in range(20):
        sino = convert_sinogram(vol_zyx[ind], spacing[0], spacing[1], spacing[2])
            
        # Normalize to [-1, 1]
        sino_min, sino_max = sino.min(), sino.max()
        print(sino_min, sino_max)
        sino = projection(sino, spacing)
        sino = convert_mu_to_hu(sino) 
        sino= np.nan_to_num(sino,nan=-1024,posinf=3071,neginf=-1024)
        print(sino.min(), sino.max())
        save_ct_volume_as_dicom(sino, spacing[::-1], Path("/home/a/ahmedlamidi/G-led/dicom"), patient_id ="hi",study_uid = generate_uid(),series_uid= generate_uid())
        print("here")
        exit(0)
        #if sino_max > sino_min:
            #sino = 2.0 * (sino - sino_min) / (sino_max - sino_min) - 1.0
                        
            # NumPy -> Torch
            
            #projection(sino, spacing)
            #exit(0)
                #print(sino.shape)  # (512, 512)
            
            #np.save("sino.npy", sino)
            #self.sinograms.append(sino)

if __name__ == "__main__":
    main("data/extra_data/1-133.dcm")

