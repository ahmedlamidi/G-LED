from pathlib import Path
import SimpleITK as sitk
import numpy as np
import imageio.v3 as iio
import astra
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
import json



def visualize(ct_slice: np.ndarray):
    
    ct_slice = ct_slice.astype(np.float32, copy=False)
    lo = np.percentile(ct_slice, 2)
    hi = np.percentile(ct_slice, 97)
    #lo, hi = [-3024.0, 1000.0]
    gt_clipped = np.clip(ct_slice, lo, hi).astype(np.float32, copy=False)
    gt_clipped = 0.2 * (1.0 + gt_clipped / 1000.0)
    vmin, vmax = float(np.min(gt_clipped)), float(np.max(gt_clipped))
    if vmax > vmin:
        gt_clipped =  (gt_clipped - vmin)  / (vmax - vmin)
        v_u8 = (gt_clipped * 255.0 + 0.5).astype(np.uint8)
        iio.imwrite("ct.png", v_u8)
    # print("done")
    

#Need to work on this to convert to different configs later
# need to do the sampling using astra
def compute_fbp_reprojection(sino_normalized, cond_indices,
                              det_count=816, num_angles=720,
                              DSO=1000, ODD=600):
    """
    Take a normalized sinogram, run FBP on known rows, then forward-project
    back to all angles to get a coarse full-sinogram estimate.

    sino_normalized: [num_angles, det_count] sinogram in [-1, 1]
    cond_indices: list of known row indices
    Returns: [num_angles, det_count] re-projected sinogram in [-1, 1]
    """
    # Extract known rows only
    sparse_sino = sino_normalized[cond_indices, :].astype(np.float32)

    # Full and sparse angle arrays
    angles_full = np.linspace(0, 2 * np.pi, num_angles, endpoint=False).astype(np.float32)
    angles_sparse = angles_full[cond_indices]

    # Reconstruction grid
    N_pix = 512
    vol_geom = astra.create_vol_geom(N_pix, N_pix)

    # --- FBP with sparse angles ---
    proj_geom_sparse = astra.create_proj_geom(
        'fanflat', 1.0, det_count, angles_sparse, DSO, ODD
    )
    rec_id = astra.data2d.create('-vol', vol_geom)
    sino_id = astra.data2d.create('-sino', proj_geom_sparse, sparse_sino)

    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    fbp_image = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)

    # --- Forward-project FBP image to full sinogram (all angles) ---
    proj_geom_full = astra.create_proj_geom(
        'fanflat', 1.0, det_count, angles_full, DSO, ODD
    )
    vol_id = astra.data2d.create('-vol', vol_geom, fbp_image)
    proj_id = astra.create_projector('line_fanflat', proj_geom_full, vol_geom)
    sino_id2, reproj_sino = astra.create_sino(vol_id, proj_id)

    # Normalize to [-1, 1]
    rmin, rmax = reproj_sino.min(), reproj_sino.max()
    if rmax > rmin:
        reproj_sino = 2.0 * (reproj_sino - rmin) / (rmax - rmin) - 1.0

    # Cleanup
    astra.data2d.delete(sino_id2)
    astra.data2d.delete(vol_id)
    astra.projector.delete(proj_id)

    return reproj_sino.astype(np.float32)


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
def convert_sinogram(ct_slice, dx, dy, dz, detector_count, angle_step):
    
    #det count should be width of a pixel
    #dx_mm which is det_spacing is dx_mm
    
    # Convert HU to attenuation coefficients first
    mu_slice = convert_hu_to_mu(ct_slice)
    
    H, W = list(mu_slice.shape)[:2]
    
    DSO = 1000
    ODD = 600  
    angles_deg = np.arange(0, 360, angle_step, dtype=np.float32)
    angles = np.deg2rad(angles_deg)  # ASTRA expects radians
    
    # generate params for the second part
    vol_geom = astra.create_vol_geom( H, W,
        -W * dx / 2.0,  W* dx/ 2.0,   # x_min, x_max
        -H * dy / 2.0,  H * dy / 2.0    # y_min, y_max
    )
        
    # Detector should cover the full object diagonal
    det_count = detector_count
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
        #get the volumes and the spacings

def load_series_from(path):
    # 	Extended: If path is a directory, read all series in it
    if path.endswith(".dcm"):
        img = sitk.ReadImage(path)
        vol_zyx = sitk.GetArrayFromImage(img)
        spacing_x, spacing_y, spacing_z = img.GetSpacing()
        return [vol_zyx, (spacing_x, spacing_y, spacing_z)]
    else:
        series_dir = Path(path)
        sitk_reader = sitk.ImageSeriesReader()
        # Get all series UIDs in the directory
        series_IDs = sitk_reader.GetGDCMSeriesIDs(str(series_dir))
        if not series_IDs:
            raise ValueError(f"No DICOM series found in directory: {series_dir}")
        series_dict = []
        for series_uid in series_IDs:
            print(series_uid)
            file_names = sitk_reader.GetGDCMSeriesFileNames(str(series_dir), series_uid)
            sitk_reader.SetFileNames(file_names)
            img = sitk_reader.Execute()
            vol_zyx = sitk.GetArrayFromImage(img)
            spacing_x, spacing_y, spacing_z = img.GetSpacing()
            series_dict.append((vol_zyx, (spacing_x, spacing_y, spacing_z)))
        return series_dict

def cache_fbp_reprojections(index_map, cond_indices, detector_count, num_angles, data_tag=None):
    """FBP re-projection of every cached sinogram for these measured rows, computed
    once; returns the cache paths in index_map's order. The folder name holds
    the view pattern (and data_tag) so stale caches aren't reused."""
    n_cond = len(cond_indices)
    max_ang = max(cond_indices)
    step = cond_indices[1] - cond_indices[0] if len(cond_indices) > 1 else 0
    fbp_tag = "fbp_" + (f"{data_tag}_" if data_tag else "") + f"n{n_cond}_step{step}_max{max_ang}"
    fbp_cache_dir = os.path.join("data", f"{fbp_tag}_cache")
    os.makedirs(fbp_cache_dir, exist_ok=True)
    print(f"Caching FBP re-projections ({fbp_tag}) in {fbp_cache_dir} ...")
    fbp_map = []
    for sino_path in tqdm(index_map, desc="FBP re-projection", mininterval=30):
        fbp_path = os.path.join(fbp_cache_dir, os.path.basename(sino_path).replace("sino_", f"{fbp_tag}_"))
        if not os.path.exists(fbp_path):
            fbp = compute_fbp_reprojection(np.load(sino_path), cond_indices,
                                           det_count=detector_count, num_angles=num_angles)
            tmp_path = fbp_path + ".tmp.npy"      # a job killed mid-write leaves no half file behind
            np.save(tmp_path, fbp)
            os.replace(tmp_path, fbp_path)
        fbp_map.append(fbp_path)
    print(f"FBP re-projections ready: {len(fbp_map)} cached")
    return fbp_map


def fit_det_spacing(sino, DSO=1000, ODD=600, lo=0.3, hi=1.3):
    """Detector pitch of a cached sinogram, for when the DICOM it came from is
    gone. A fan-beam sinogram measures every line twice: view theta at detector
    coordinate u is view theta + 180 deg - 2 atan(u / (DSO + ODD)) at -u. The
    two readings only agree at the true pitch, so it is the pitch with the
    smallest mismatch (coarse grid, then golden section)."""
    H, W = sino.shape
    sino = sino.astype(np.float64)
    centred, rows, cols = np.arange(W) - (W - 1) / 2, np.arange(H)[:, None], np.arange(W - 1, -1, -1)[None, :]

    def mismatch(dx):
        shift = (np.pi - 2 * np.arctan(centred * dx / (DSO + ODD))) * H / (2 * np.pi)
        low = np.floor(shift).astype(int)
        frac = shift - low
        partner = (1 - frac) * sino[(rows + low) % H, cols] + frac * sino[(rows + low + 1) % H, cols]
        return np.mean((sino - partner) ** 2)

    grid = np.linspace(lo, hi, 41)
    k = int(np.argmin([mismatch(dx) for dx in grid]))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = mismatch(c), mismatch(d)
    while b - a > 1e-4:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = mismatch(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = mismatch(d)
    return float((a + b) / 2)


#correct so far 
#vol_zyx , spacing = load_series_from("data/Dataset")
#sinograms = [convert_sinogram(slice, spacing[0], spacing[1], spacing[2]) for slice in vol_zyx]

class dicom_dataset(Dataset):

    def __init__(self, data_path="data/Dataset",
                 detector_count=816,
                 angle_step=(360/720),
                 cache_dir="data/sino_cache",
                 cond_indices=None,
                 return_spacing=False):

        # Sanitize data_path for use in directory name
        def sanitize_path(path):
            return path.strip().replace("/", "_").replace("\\", "_")

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.index_map = []  # list of cache file paths
        self.cond_indices = cond_indices

        num_angles = int(360 / angle_step)

        # return_spacing adds the detector pitch of each sinogram (the series'
        # pixel spacing, see convert_sinogram) to every item; the fan-beam
        # physics loss needs it.
        self.return_spacing = return_spacing
        self.det_spacing = []

        # The manifest lists the cached sinograms and their detector pitch, so
        # a start with a complete cache does not read every DICOM series again.
        tag = "" if data_path == "data/Dataset" else f"{sanitize_path(data_path)}_"
        manifest_path = os.path.join(cache_dir, f"manifest_{tag}d{detector_count}_a{angle_step:.4f}.json")
        n_source_files = 1 if os.path.isfile(data_path) else sum(len(f) for _, _, f in os.walk(data_path))
        manifest = None
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest["n_source_files"] != n_source_files or \
                    not all(os.path.exists(os.path.join(cache_dir, name)) for name, _ in manifest["slices"]):
                manifest = None

        if manifest is not None:
            print(f"Sinogram cache complete ({manifest_path}), DICOM read skipped")
            self.index_map = [os.path.join(cache_dir, name) for name, _ in manifest["slices"]]
            self.det_spacing = [spacing for _, spacing in manifest["slices"]]
        else:
            total_series = load_series_from(data_path)
            if data_path.endswith(".dcm"):
                total_series = [total_series]

            for s_idx, series in enumerate(total_series):
                vol_zyx, spacing = series
                for ind in range(len(vol_zyx)):
                    cache_filename = f"sino_{tag}s{s_idx}_i{ind}_d{detector_count}_a{angle_step:.4f}.npy"
                    cache_path = os.path.join(cache_dir, cache_filename)

                    # Only compute if not already cached
                    if not os.path.exists(cache_path):
                        sino = convert_sinogram(
                            vol_zyx[ind],
                            spacing[0], spacing[1], spacing[2],
                            detector_count, angle_step
                        )
                        sino_min, sino_max = sino.min(), sino.max()
                        if sino_max > sino_min:
                            sino = 2.0 * (sino - sino_min) / (sino_max - sino_min) - 1.0
                        np.save(cache_path, sino.astype(np.float32))
                        print(f"Cached {cache_path}")

                    self.index_map.append(cache_path)
                    self.det_spacing.append(float(spacing[0]))

            with open(manifest_path, "w") as f:
                json.dump({"n_source_files": n_source_files,
                           "slices": [[os.path.basename(path), spacing]
                                      for path, spacing in zip(self.index_map, self.det_spacing)]}, f)

        # Cache FBP re-projections if cond_indices provided
        self.fbp_map = []
        if cond_indices is not None:
            self.fbp_map = cache_fbp_reprojections(self.index_map, cond_indices, detector_count, num_angles,
                                                   None if data_path == "data/Dataset" else sanitize_path(data_path))

        print(f"Dataset ready: {len(self.index_map)} sinograms")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        # Load single sinogram on demand — fast numpy load
        sino = np.load(self.index_map[index])                          # [H, W]
        sino = torch.from_numpy(sino).float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

        item = (sino,)
        if self.fbp_map:
            fbp = np.load(self.fbp_map[index])                          # [H, W]
            fbp = torch.from_numpy(fbp).float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
            item += (fbp,)
        if self.return_spacing:
            item += (torch.tensor(self.det_spacing[index], dtype=torch.float32),)
        return item if len(item) > 1 else sino
    


class baselines_split_dataset(Dataset):
    """The sinograms of one split of baselines/prepare_data.py, so SD-Flow trains on
    exactly the slices the baselines train on. They are dicom_dataset's own
    cached sinograms; which rows are measured does not change them, so the
    slices.json of any prepared setting works. Items are dicom_dataset's with
    return_spacing: (sinogram, FBP re-projection, detector pitch).

    The detector pitch is the series' pixel spacing: from a DICOM header of the
    series when the data root of the split is still there, else fitted to a
    sinogram of the series (fit_det_spacing). Found once, kept beside slices.json.
    """

    def __init__(self, slices_json, cond_indices, detector_count=816, angle_step=(360/720)):
        with open(slices_json) as f:
            slices = json.load(f)
        self.index_map = [s["sino"] for s in slices]
        missing = [path for path in self.index_map if not os.path.exists(path)]
        if missing:
            raise SystemExit(f"{len(missing)} of {len(slices)} sinograms of {slices_json} are missing, "
                             f"e.g. {missing[0]}; run from the repo root, after python -m baselines.prepare_data")

        spacing_json = os.path.join(os.path.dirname(slices_json), "det_spacing.json")
        spacing = {}
        if os.path.exists(spacing_json):
            with open(spacing_json) as f:
                spacing = json.load(f)
        split_json = os.path.join(os.path.dirname(os.path.dirname(slices_json)), "split.json")
        data_root = None
        if os.path.exists(split_json):
            with open(split_json) as f:
                data_root = json.load(f).get("data_root")
        series_slices = {}
        for s in slices:
            series_slices.setdefault(os.path.join(s["patient"], s["series"]), []).append(s)
        for series, members in series_slices.items():
            if series in spacing:
                continue
            folder = os.path.join(data_root, series) if data_root else None
            dcm = sorted(f for f in os.listdir(folder) if f.lower().endswith(".dcm")) \
                if folder and os.path.isdir(folder) else []
            if dcm:
                reader = sitk.ImageFileReader()
                reader.SetFileName(os.path.join(folder, dcm[0]))
                reader.ReadImageInformation()
                spacing[series] = {"det_spacing": float(reader.GetSpacing()[0]), "source": "dicom header"}
            else:
                middle = members[len(members) // 2]["sino"]
                spacing[series] = {"det_spacing": fit_det_spacing(np.load(middle)), "source": "fitted to " + middle}
            print(f"detector pitch {spacing[series]['det_spacing']:.4f} ({spacing[series]['source']}): {series}")
            with open(spacing_json, "w") as f:
                json.dump(spacing, f, indent=1)
        self.det_spacing = [spacing[os.path.join(s["patient"], s["series"])]["det_spacing"] for s in slices]

        self.fbp_map = cache_fbp_reprojections(self.index_map, cond_indices, detector_count,
                                               int(360 / angle_step))
        print(f"Dataset ready: {len(self.index_map)} sinograms from {slices_json}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        sino = torch.from_numpy(np.load(self.index_map[index])).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        fbp = torch.from_numpy(np.load(self.fbp_map[index])).float().unsqueeze(0).unsqueeze(0)
        return sino, fbp, torch.tensor(self.det_spacing[index], dtype=torch.float32)


class Efficient_LIDC_DicomDataset(Dataset):
    def __init__(self, root_dir="data/LIDC-IDRI", Interpolate=False):
        """
        index_map = [
            (patient_id, study_id, series_folder, dicom_path),
            ...
        ]
        """
        self.root_dir = root_dir
        self.index_map = []
        self.Interpolate = Interpolate

        for patient in sorted(os.listdir(root_dir)):
            patient_path = os.path.join(root_dir, patient)
            if not os.path.isdir(patient_path):
                continue

            for study in sorted(os.listdir(patient_path)):
                study_path = os.path.join(patient_path, study)
                if not os.path.isdir(study_path):
                    continue

                # Series folder name is arbitrary
                for series in sorted(os.listdir(study_path)):
                    series_path = os.path.join(study_path, series)
                    if not os.path.isdir(series_path):
                        continue

                    dicom_files = [
                        f for f in os.listdir(series_path)
                        if f.lower().endswith(".dcm")
                    ]

                    for dcm in sorted(dicom_files):
                        self.index_map.append((
                            patient,
                            study,
                            series,
                            os.path.join(series_path, dcm)
                        ))

    def __len__(self):
        return len(self.index_map)

    # def __getitem__(self, index):
    #     patient_id, study_id, series_id, dicom_path = self.index_map[index]


    #     #print(dicom_path)
    #    # exit(0)
    #     # ---- Load single DICOM slice ----
    #     vol_zyx, spacing = load_series_from(dicom_path)

    #     # If load_series_from returns a single slice wrapped as volume

    #     sino = convert_sinogram(vol_zyx[0], spacing[0], spacing[1], spacing[2])
        
    #     # Normalize to [-1, 1]
    #     sino_min, sino_max = sino.min(), sino.max()
    #     if sino_max > sino_min:
    #         sino = 2.0 * (sino - sino_min) / (sino_max - sino_min) - 1.0
                    
    #     # NumPy -> Torch
    #     # sino = torch.from_numpy(sino).float()
        
    #     # sino = sino.unsqueeze(0).unsqueeze(0)
        
        
    #     sino = torch.from_numpy(sino).float().unsqueeze(0).unsqueeze(0)
    #     return sino

        # Resize
        # sino = F.interpolate(   
        #     sino,
        #     size=(1024, 1024),
        #     mode="nearest"   # correct for sinograms
        # )


if __name__ == '__main__':
	# Create ground truth folder if it doesn't exist
	ground_truth_dir = "output/feb_19_720_820_model/ground_truth"
	os.makedirs(ground_truth_dir, exist_ok=True)
	
	# Use dicom_dataset to load data
	dset = dicom_dataset(Interpolate=True, detector_count= 816, angle_step=(360/720), data_path="data/test_data")
	
	print(f"Total samples in dataset: {len(dset)}")
	
	# Save first 20 sinograms as batch0.npy through batch19.npy
	print("Saving first 20 sinograms as batch files...")
	for i in range(min(100, len(dset))):
		# Get sinogram and squeeze to [H, W]
		sino_np = dset.sinograms_torch[i].squeeze().numpy()
		
		# Save as batchX.npy
		filename = f"batch{i}.npy"
		filepath = os.path.join(ground_truth_dir, filename)
		np.save(filepath, sino_np)
		print(f"Saved {filename} with shape {sino_np.shape}")
	
	print(f"Done! Saved 20 batch files to {ground_truth_dir}")




