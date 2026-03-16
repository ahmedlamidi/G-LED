import sys
import os
import numpy as np
import astra
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from dicom_preprocess import load_series_from, convert_hu_to_mu


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


def build_sparse_geometry(vol_geom, full_angles, cond_indices, dx, detector_count):
    """Build ASTRA geometry using only the selected sparse angles."""
    DSO = 1000
    ODD = 600
    sparse_angles = full_angles[cond_indices]
    proj_geom = astra.create_proj_geom(
        'fanflat', dx, detector_count, sparse_angles, DSO, ODD
    )
    return proj_geom


def compare_slice(mu_slice, vol_geom, proj_geom_full, full_angles,
                  cond_indices, dx, detector_count, save_dir, slice_idx):
    """Compare full vs sparse FBP for a single slice."""
    # Full sinogram and FBP
    sino_full = forward_project(mu_slice, vol_geom, proj_geom_full)
    recon_full = fbp_reconstruct(sino_full, vol_geom, proj_geom_full)

    # Sparse sinogram (select only conditioned rows)
    sino_sparse = sino_full[cond_indices, :]

    # Sparse FBP
    proj_geom_sparse = build_sparse_geometry(
        vol_geom, full_angles, cond_indices, dx, detector_count
    )
    recon_sparse = fbp_reconstruct(sino_sparse, vol_geom, proj_geom_sparse)

    # -- Sinogram metrics --
    # Pad sparse sinogram back to full size for comparison (zeros for missing rows)
    sino_padded = np.zeros_like(sino_full)
    sino_padded[cond_indices, :] = sino_sparse
    # Only compare on known rows for sinogram SSIM/PSNR (they're identical),
    # so instead compare full vs padded to show information loss
    sino_data_range = sino_full.max() - sino_full.min()
    sino_ssim = ssim(sino_full, sino_padded, data_range=sino_data_range)
    sino_psnr = psnr(sino_full, sino_padded, data_range=sino_data_range)

    # -- Reconstruction metrics --
    recon_data_range = recon_full.max() - recon_full.min()
    recon_ssim = ssim(recon_full, recon_sparse, data_range=recon_data_range)
    recon_psnr = psnr(recon_full, recon_sparse, data_range=recon_data_range)

    # -- Visualization --
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Sinograms
    axes[0, 0].imshow(sino_full, cmap='gray', aspect='auto')
    axes[0, 0].set_title('Full Sinogram (720 angles)')
    axes[0, 1].imshow(sino_padded, cmap='gray', aspect='auto')
    axes[0, 1].set_title(f'Sparse Sinogram ({len(cond_indices)} angles)')
    axes[0, 2].imshow(np.abs(sino_full - sino_padded), cmap='hot', aspect='auto')
    axes[0, 2].set_title(f'Sino Diff (SSIM={sino_ssim:.4f}, PSNR={sino_psnr:.2f})')

    # Row 2: Reconstructions
    vmin, vmax = recon_full.min(), recon_full.max()
    axes[1, 0].imshow(recon_full, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('FBP Full (720 angles)')
    axes[1, 1].imshow(recon_sparse, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'FBP Sparse ({len(cond_indices)} angles)')
    axes[1, 2].imshow(np.abs(recon_full - recon_sparse), cmap='hot')
    axes[1, 2].set_title(f'Recon Diff (SSIM={recon_ssim:.4f}, PSNR={recon_psnr:.2f})')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'slice_{slice_idx}.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)

    return {
        'sino_ssim': sino_ssim, 'sino_psnr': sino_psnr,
        'recon_ssim': recon_ssim, 'recon_psnr': recon_psnr,
    }


def main():
    data_path = "data/test_data"
    detector_count = 816
    angle_step = 360 / 720
    save_dir = "Comparsion/results"
    os.makedirs(save_dir, exist_ok=True)

    # Same mask as train_diff: every 10th row in first 75%
    total_angles = 720
    cutoff = int(total_angles * 0.75)
    cond_indices = [i for i in range(total_angles) if i % 10 == 0 and i < cutoff]
    print(f"Mask: {len(cond_indices)} known angles out of {total_angles} "
          f"(first {cutoff}, every 10th)")

    # Load test data
    total_series = load_series_from(data_path)
    all_metrics = []

    for s_idx, series in enumerate(total_series):
        vol_zyx, spacing = series
        dx, dy, dz = spacing

        # Build full geometry once per series
        H, W = vol_zyx[0].shape[:2]
        vol_geom, proj_geom_full, full_angles = build_geometry(
            H, W, dx, dy, detector_count, angle_step
        )

        for ind in range(len(vol_zyx)):
            slice_idx = sum(len(total_series[j][0]) for j in range(s_idx)) + ind
            mu_slice = convert_hu_to_mu(vol_zyx[ind])

            metrics = compare_slice(
                mu_slice, vol_geom, proj_geom_full, full_angles,
                cond_indices, dx, detector_count, save_dir, slice_idx
            )
            all_metrics.append(metrics)

            print(f"Slice {slice_idx}: "
                  f"Sino SSIM={metrics['sino_ssim']:.4f} PSNR={metrics['sino_psnr']:.2f} | "
                  f"Recon SSIM={metrics['recon_ssim']:.4f} PSNR={metrics['recon_psnr']:.2f}")

    # Summary
    if all_metrics:
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print(f"\n--- Average over {len(all_metrics)} slices ---")
        print(f"Sinogram  SSIM={avg['sino_ssim']:.4f}  PSNR={avg['sino_psnr']:.2f}")
        print(f"Recon     SSIM={avg['recon_ssim']:.4f}  PSNR={avg['recon_psnr']:.2f}")

        np.savez(os.path.join(save_dir, 'metrics.npz'), **{
            k: [m[k] for m in all_metrics] for k in all_metrics[0]
        })


if __name__ == '__main__':
    main()
