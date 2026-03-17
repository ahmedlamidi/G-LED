import sys
import os
import csv
import numpy as np
import astra
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from dicom_preprocess import load_series_from, convert_hu_to_mu


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
    sino_full = forward_project(mu_slice, vol_geom, proj_geom_full)
    recon_full_mu = fbp_reconstruct(sino_full, vol_geom, proj_geom_full)

    sino_limited = sino_full[cond_indices, :]
    proj_geom_limited = build_sparse_geometry(
        full_angles, cond_indices, dx, detector_count
    )

    recon_sart = sart_reconstruct(sino_limited, vol_geom, proj_geom_limited)

    recon_data_range = recon_full_mu.max() - recon_full_mu.min()
    sart_ssim = ssim(recon_full_mu, recon_sart, data_range=recon_data_range)
    sart_psnr = psnr(recon_full_mu, recon_sart, data_range=recon_data_range)

    if slice_idx < 10:
        vmin, vmax = recon_full_mu.min(), recon_full_mu.max()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(recon_full_mu, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0].set_title('Full FBP (reference)')

        axes[1].imshow(recon_sart, cmap='gray', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'SART\nSSIM={sart_ssim:.4f}, PSNR={sart_psnr:.2f}')

        axes[2].imshow(np.abs(recon_full_mu - recon_sart), cmap='hot')
        axes[2].set_title('Error')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'slice_{slice_idx}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)

    return {'sart_ssim': sart_ssim, 'sart_psnr': sart_psnr}


def run_setting(cutoff_pct, detector_count, angle_step, total_series):
    total_angles = 720
    cutoff = int(total_angles * cutoff_pct / 100)
    cond_indices = list(range(cutoff))
    n_known = len(cond_indices)

    print(f"\n{'='*80}")
    print(f"Limited view: first {cutoff_pct}% = {n_known} of {total_angles} angles")
    print(f"{'='*80}")

    save_dir = f"Comparsion/SART_limited_{cutoff_pct}pct_{n_known}of{total_angles}"
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

            print(f"  Slice {len(all_metrics)-1}: SART SSIM={metrics['sart_ssim']:.4f} PSNR={metrics['sart_psnr']:.2f}")

    if all_metrics:
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print(f"\n--- {cutoff_pct}% Average over {len(all_metrics)} slices ---")
        print(f"  SART: SSIM={avg['sart_ssim']:.4f}  PSNR={avg['sart_psnr']:.2f}")

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

    cutoff_pcts = [6.25, 12.5, 25, 37.5, 50]
    summary = {}

    for pct in cutoff_pcts:
        avg = run_setting(pct, detector_count, angle_step, total_series)
        if avg:
            summary[pct] = avg

    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON — Limited View SART")
    print(f"{'='*80}")
    print(f"{'View %':<10} {'Angles':<8} {'SSIM':<10} {'PSNR':<10}")
    print("-" * 40)
    for pct in cutoff_pcts:
        if pct in summary:
            avg = summary[pct]
            n = int(720 * pct / 100)
            print(f"{pct:<10} {n:<8} {avg['sart_ssim']:<10.4f} {avg['sart_psnr']:<10.2f}")


if __name__ == '__main__':
    main()
