import sys
import os
import numpy as np
import astra
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from dicom_preprocess import load_series_from, convert_hu_to_mu


def convert_mu_to_hu(mu):
    """Inverse of convert_hu_to_mu: HU = (mu / 0.02 - 1) * 1000."""
    return (mu / 0.02 - 1.0) * 1000.0


def build_geometry(H, W, dx, dy, detector_count, angle_step):
    """Build ASTRA fan-beam geometry matching dicom_preprocess.convert_sinogram."""
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
    """Create sinogram from attenuation image."""
    slice2d = np.ascontiguousarray(mu_slice, dtype=np.float32)
    sid = astra.data2d.create('-vol', vol_geom, slice2d)
    proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
    sino_id, sino = astra.create_sino(sid, proj_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(sid)
    astra.projector.delete(proj_id)
    return sino


def fbp_reconstruct(sinogram, vol_geom, proj_geom):
    """FBP reconstruction from a sinogram."""
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
    """Build ASTRA geometry using only the selected sparse angles."""
    DSO = 1000
    ODD = 600
    sparse_angles = full_angles[cond_indices]
    proj_geom = astra.create_proj_geom(
        'fanflat', dx, detector_count, sparse_angles, DSO, ODD
    )
    return proj_geom


def compare_slice(ct_slice_hu, mu_slice, vol_geom, proj_geom_full, full_angles,
                  cond_indices, dx, detector_count, save_dir, slice_idx):
    """Compare full vs sparse FBP for a single slice in sinogram, recon, and DICOM space."""
    # Full sinogram and FBP (in mu space)
    sino_full = forward_project(mu_slice, vol_geom, proj_geom_full)
    recon_full_mu = fbp_reconstruct(sino_full, vol_geom, proj_geom_full)

    # Sparse sinogram (select only conditioned rows)
    sino_sparse = sino_full[cond_indices, :]

    # Sparse FBP (in mu space)
    proj_geom_sparse = build_sparse_geometry(
        full_angles, cond_indices, dx, detector_count
    )
    recon_sparse_mu = fbp_reconstruct(sino_sparse, vol_geom, proj_geom_sparse)

    # Convert reconstructions to HU (DICOM space)
    recon_full_hu = convert_mu_to_hu(recon_full_mu)
    recon_sparse_hu = convert_mu_to_hu(recon_sparse_mu)

    # -- Sinogram metrics --
    sino_padded = np.zeros_like(sino_full)
    sino_padded[cond_indices, :] = sino_sparse
    sino_data_range = sino_full.max() - sino_full.min()
    sino_ssim = ssim(sino_full, sino_padded, data_range=sino_data_range)
    sino_psnr = psnr(sino_full, sino_padded, data_range=sino_data_range)

    # -- Reconstruction metrics (mu space) --
    recon_data_range = recon_full_mu.max() - recon_full_mu.min()
    recon_ssim = ssim(recon_full_mu, recon_sparse_mu, data_range=recon_data_range)
    recon_psnr = psnr(recon_full_mu, recon_sparse_mu, data_range=recon_data_range)

    # -- DICOM space metrics (HU): compare full FBP vs sparse FBP --
    hu_data_range = recon_full_hu.max() - recon_full_hu.min()
    dicom_ssim = ssim(recon_full_hu, recon_sparse_hu, data_range=hu_data_range)
    dicom_psnr = psnr(recon_full_hu, recon_sparse_hu, data_range=hu_data_range)

    # -- Visualization (3 rows) --
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Row 1: Sinograms
    axes[0, 0].imshow(sino_full, cmap='gray', aspect='auto')
    axes[0, 0].set_title('Full Sinogram (720 angles)')
    axes[0, 1].imshow(sino_padded, cmap='gray', aspect='auto')
    axes[0, 1].set_title(f'Sparse Sinogram ({len(cond_indices)} angles)')
    axes[0, 2].imshow(np.abs(sino_full - sino_padded), cmap='hot', aspect='auto')
    axes[0, 2].set_title(f'Sino Diff (SSIM={sino_ssim:.4f}, PSNR={sino_psnr:.2f})')

    # Row 2: FBP Reconstructions (mu space)
    vmin, vmax = recon_full_mu.min(), recon_full_mu.max()
    axes[1, 0].imshow(recon_full_mu, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('FBP Full (720 angles)')
    axes[1, 1].imshow(recon_sparse_mu, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'FBP Sparse ({len(cond_indices)} angles)')
    axes[1, 2].imshow(np.abs(recon_full_mu - recon_sparse_mu), cmap='hot')
    axes[1, 2].set_title(f'Recon Diff (SSIM={recon_ssim:.4f}, PSNR={recon_psnr:.2f})')

    # Row 3: DICOM space (HU) — full FBP vs sparse FBP
    hu_vmin, hu_vmax = -1024.0, 1000.0
    axes[2, 0].imshow(recon_full_hu, cmap='gray', vmin=hu_vmin, vmax=hu_vmax)
    axes[2, 0].set_title('Full FBP → HU')
    axes[2, 1].imshow(recon_sparse_hu, cmap='gray', vmin=hu_vmin, vmax=hu_vmax)
    axes[2, 1].set_title(f'Sparse FBP → HU')
    axes[2, 2].imshow(np.abs(recon_full_hu - recon_sparse_hu), cmap='hot')
    axes[2, 2].set_title(f'HU Diff (SSIM={dicom_ssim:.4f}, PSNR={dicom_psnr:.2f})')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    if slice_idx < 10:
        fig.savefig(os.path.join(save_dir, f'slice_{slice_idx}.png'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)

    return {
        'sino_ssim': sino_ssim, 'sino_psnr': sino_psnr,
        'recon_ssim': recon_ssim, 'recon_psnr': recon_psnr,
        'dicom_ssim': dicom_ssim, 'dicom_psnr': dicom_psnr,
    }


def run_setting(cutoff_pct, data_path, detector_count, angle_step, total_series):
    """Run FBP comparison for a limited-view setting (first cutoff_pct% of angles)."""
    total_angles = 720
    cutoff = int(total_angles * cutoff_pct / 100)
    cond_indices = list(range(cutoff))
    n_known = len(cond_indices)

    print(f"\n{'='*80}")
    print(f"Limited view: first {cutoff_pct}% = {n_known} of {total_angles} angles")
    print(f"{'='*80}")

    save_dir = f"Comparsion/FBP_limited_{cutoff_pct}pct_{n_known}of{total_angles}"
    os.makedirs(save_dir, exist_ok=True)

    all_metrics = []
    global_slice_idx = 0

    for s_idx, series in enumerate(total_series):
        vol_zyx, spacing = series
        dx, dy, dz = spacing

        H, W = vol_zyx[0].shape[:2]
        vol_geom, proj_geom_full, full_angles = build_geometry(
            H, W, dx, dy, detector_count, angle_step
        )

        for ind in range(len(vol_zyx)):
            ct_slice_hu = vol_zyx[ind].astype(np.float32)
            mu_slice = convert_hu_to_mu(ct_slice_hu)

            metrics = compare_slice(
                ct_slice_hu, mu_slice, vol_geom, proj_geom_full, full_angles,
                cond_indices, dx, detector_count, save_dir, global_slice_idx
            )
            all_metrics.append(metrics)

            print(f"  Slice {global_slice_idx}: "
                  f"Recon SSIM={metrics['recon_ssim']:.4f} PSNR={metrics['recon_psnr']:.2f} | "
                  f"DICOM SSIM={metrics['dicom_ssim']:.4f} PSNR={metrics['dicom_psnr']:.2f}")

            global_slice_idx += 1

    # Summary
    if all_metrics:
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print(f"\n--- {cutoff_pct}% Average over {len(all_metrics)} slices ---")
        print(f"Sinogram       SSIM={avg['sino_ssim']:.4f}  PSNR={avg['sino_psnr']:.2f}")
        print(f"Recon (mu)     SSIM={avg['recon_ssim']:.4f}  PSNR={avg['recon_psnr']:.2f}")
        print(f"DICOM full-vs-limited SSIM={avg['dicom_ssim']:.4f}  PSNR={avg['dicom_psnr']:.2f}")

        import csv
        csv_path = os.path.join(save_dir, 'metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['slice'] + list(all_metrics[0].keys()))
            writer.writeheader()
            for i, m in enumerate(all_metrics):
                writer.writerow({'slice': i, **m})

    return avg if all_metrics else None


def main():
    data_path = "data/test_data"
    detector_count = 816
    angle_step = 360 / 720

    # Load test data once
    total_series = load_series_from(data_path)

    cutoff_pcts = [100]
    summary = {}

    for pct in cutoff_pcts:
        avg = run_setting(pct, data_path, detector_count, angle_step, total_series)
        if avg:
            summary[pct] = avg

    # Print final comparison table
    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON — Limited View FBP")
    print(f"{'='*80}")
    print(f"{'View %':<10} {'Angles':<10} {'Recon SSIM':<12} {'Recon PSNR':<12} {'DICOM SSIM':<12} {'DICOM PSNR':<12}")
    print("-" * 68)
    for pct in cutoff_pcts:
        if pct in summary:
            avg = summary[pct]
            n = int(720 * pct / 100)
            print(f"{pct:<10} {n:<10} {avg['recon_ssim']:<12.4f} {avg['recon_psnr']:<12.2f} "
                  f"{avg['dicom_ssim']:<12.4f} {avg['dicom_psnr']:<12.2f}")


if __name__ == '__main__':
    main()
