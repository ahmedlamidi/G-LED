import sys
import os
import numpy as np
import astra
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import denoise_tv_chambolle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from dicom_preprocess import load_series_from, convert_hu_to_mu


def convert_mu_to_hu(mu):
    return (mu / 0.02 - 1.0) * 1000.0


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


def sart_reconstruct(sinogram, vol_geom, proj_geom, iterations=200):
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict('SART_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iterations)
    recon = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)
    return recon


def sart_tv_reconstruct(sinogram, vol_geom, proj_geom,
                         outer_iterations=20, sart_iters_per_step=10,
                         tv_weight=0.002):
    """SART with TV regularization: alternate SART steps with TV denoising."""
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict('SART_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id

    alg_id = astra.algorithm.create(cfg)

    for _ in range(outer_iterations):
        # SART update steps
        astra.algorithm.run(alg_id, sart_iters_per_step)

        # TV denoising step
        recon = astra.data2d.get(rec_id)
        recon = denoise_tv_chambolle(recon, weight=tv_weight)
        astra.data2d.store(rec_id, recon.astype(np.float32))

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


def compare_slice(mu_slice, vol_geom, proj_geom_full, full_angles,
                  cond_indices, dx, detector_count, save_dir, slice_idx):
    """Compare FBP vs SART vs SART+TV for limited-view reconstruction."""
    # Full sinogram and full FBP (reference)
    sino_full = forward_project(mu_slice, vol_geom, proj_geom_full)
    recon_full_mu = fbp_reconstruct(sino_full, vol_geom, proj_geom_full)

    # Limited sinogram
    sino_limited = sino_full[cond_indices, :]
    proj_geom_limited = build_sparse_geometry(
        full_angles, cond_indices, dx, detector_count
    )

    # Three reconstruction methods on limited data
    recon_fbp = fbp_reconstruct(sino_limited, vol_geom, proj_geom_limited)
    recon_sart = sart_reconstruct(sino_limited, vol_geom, proj_geom_limited)
    recon_tv = sart_tv_reconstruct(sino_limited, vol_geom, proj_geom_limited)

    # Metrics: each vs full FBP reference
    recon_data_range = recon_full_mu.max() - recon_full_mu.min()

    fbp_ssim = ssim(recon_full_mu, recon_fbp, data_range=recon_data_range)
    fbp_psnr = psnr(recon_full_mu, recon_fbp, data_range=recon_data_range)

    sart_ssim = ssim(recon_full_mu, recon_sart, data_range=recon_data_range)
    sart_psnr = psnr(recon_full_mu, recon_sart, data_range=recon_data_range)

    tv_ssim = ssim(recon_full_mu, recon_tv, data_range=recon_data_range)
    tv_psnr = psnr(recon_full_mu, recon_tv, data_range=recon_data_range)

    # Visualization
    if slice_idx < 10:
        vmin, vmax = recon_full_mu.min(), recon_full_mu.max()
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))

        # Row 1: Reconstructions
        axes[0, 0].imshow(recon_full_mu, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Full FBP (reference)')

        axes[0, 1].imshow(recon_fbp, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f'Limited FBP\nSSIM={fbp_ssim:.4f}')

        axes[0, 2].imshow(recon_sart, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title(f'SART\nSSIM={sart_ssim:.4f}')

        axes[0, 3].imshow(recon_tv, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 3].set_title(f'SART+TV\nSSIM={tv_ssim:.4f}')

        # Row 2: Error maps
        axes[1, 0].imshow(sino_full, cmap='gray', aspect='auto')
        axes[1, 0].set_title('Full Sinogram')

        axes[1, 1].imshow(np.abs(recon_full_mu - recon_fbp), cmap='hot')
        axes[1, 1].set_title(f'FBP Error (PSNR={fbp_psnr:.2f})')

        axes[1, 2].imshow(np.abs(recon_full_mu - recon_sart), cmap='hot')
        axes[1, 2].set_title(f'SART Error (PSNR={sart_psnr:.2f})')

        axes[1, 3].imshow(np.abs(recon_full_mu - recon_tv), cmap='hot')
        axes[1, 3].set_title(f'SART+TV Error (PSNR={tv_psnr:.2f})')

        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'slice_{slice_idx}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)

    return {
        'fbp_ssim': fbp_ssim, 'fbp_psnr': fbp_psnr,
        'sart_ssim': sart_ssim, 'sart_psnr': sart_psnr,
        'tv_ssim': tv_ssim, 'tv_psnr': tv_psnr,
    }


def run_setting(cutoff_pct, detector_count, angle_step, total_series):
    total_angles = 720
    cutoff = int(total_angles * cutoff_pct / 100)
    cond_indices = list(range(cutoff))
    n_known = len(cond_indices)

    print(f"\n{'='*80}")
    print(f"Limited view: first {cutoff_pct}% = {n_known} of {total_angles} angles")
    print(f"{'='*80}")

    save_dir = f"Comparsion/SART_TV_limited_{cutoff_pct}pct_{n_known}of{total_angles}"
    os.makedirs(save_dir, exist_ok=True)

    all_metrics = []

    for s_idx, series in enumerate(total_series):
        vol_zyx, spacing = series
        dx, dy, dz = spacing

        H, W = vol_zyx[0].shape[:2]
        vol_geom, proj_geom_full, full_angles = build_geometry(
            H, W, dx, dy, detector_count, angle_step
        )

        for ind in range(len(vol_zyx)):
            mu_slice = convert_hu_to_mu(vol_zyx[ind].astype(np.float32))

            metrics = compare_slice(
                mu_slice, vol_geom, proj_geom_full, full_angles,
                cond_indices, dx, detector_count, save_dir, len(all_metrics)
            )
            all_metrics.append(metrics)

            print(f"  Slice {len(all_metrics)-1}: "
                  f"FBP={metrics['fbp_ssim']:.4f} | "
                  f"SART={metrics['sart_ssim']:.4f} | "
                  f"SART+TV={metrics['tv_ssim']:.4f}")

    if all_metrics:
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print(f"\n--- {cutoff_pct}% Average over {len(all_metrics)} slices ---")
        print(f"  FBP:     SSIM={avg['fbp_ssim']:.4f}  PSNR={avg['fbp_psnr']:.2f}")
        print(f"  SART:    SSIM={avg['sart_ssim']:.4f}  PSNR={avg['sart_psnr']:.2f}")
        print(f"  SART+TV: SSIM={avg['tv_ssim']:.4f}  PSNR={avg['tv_psnr']:.2f}")

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

    total_series = load_series_from(data_path)

    cutoff_pcts = [12.5, 25, 50, 75]
    summary = {}

    for pct in cutoff_pcts:
        avg = run_setting(pct, detector_count, angle_step, total_series)
        if avg:
            summary[pct] = avg

    # Final comparison table
    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON — Limited View: FBP vs SART vs SART+TV")
    print(f"{'='*80}")
    print(f"{'View %':<10} {'Angles':<8} "
          f"{'FBP SSIM':<10} {'FBP PSNR':<10} "
          f"{'SART SSIM':<11} {'SART PSNR':<11} "
          f"{'TV SSIM':<10} {'TV PSNR':<10}")
    print("-" * 90)
    for pct in cutoff_pcts:
        if pct in summary:
            avg = summary[pct]
            n = int(720 * pct / 100)
            print(f"{pct:<10} {n:<8} "
                  f"{avg['fbp_ssim']:<10.4f} {avg['fbp_psnr']:<10.2f} "
                  f"{avg['sart_ssim']:<11.4f} {avg['sart_psnr']:<11.2f} "
                  f"{avg['tv_ssim']:<10.4f} {avg['tv_psnr']:<10.2f}")


if __name__ == '__main__':
    main()
