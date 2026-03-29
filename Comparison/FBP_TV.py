import sys
import os
import csv
import numpy as np
import astra
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import denoise_tv_chambolle

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


def fbp_tv_reconstruct(sinogram, vol_geom, proj_geom, tv_weight=0.002):
    """FBP followed by TV denoising."""
    recon = fbp_reconstruct(sinogram, vol_geom, proj_geom)
    recon = denoise_tv_chambolle(recon, weight=tv_weight)
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

    recon_tv = fbp_tv_reconstruct(sino_limited, vol_geom, proj_geom_limited)

    recon_data_range = recon_full_mu.max() - recon_full_mu.min()
    tv_ssim = ssim(recon_full_mu, recon_tv, data_range=recon_data_range)
    tv_psnr = psnr(recon_full_mu, recon_tv, data_range=recon_data_range)

    if slice_idx < 10:
        vmin, vmax = recon_full_mu.min(), recon_full_mu.max()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(recon_full_mu, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0].set_title('Full FBP (reference)')

        axes[1].imshow(recon_tv, cmap='gray', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'FBP+TV\nSSIM={tv_ssim:.4f}, PSNR={tv_psnr:.2f}')

        axes[2].imshow(np.abs(recon_full_mu - recon_tv), cmap='hot')
        axes[2].set_title('Error')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'slice_{slice_idx}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)

    return {'tv_ssim': tv_ssim, 'tv_psnr': tv_psnr}


def run_setting(detector_count, angle_step, total_series,
                cutoff_pct=None, step=None):
    """
    cutoff_pct only : contiguous first N% of angles  (limited-view)
    step only       : every Nth angle within first 75%  (sparse)
    both            : every Nth angle within first cutoff_pct%  (sparse + limited-view)
    """
    total_angles = 720
    cutoff = int(total_angles * (cutoff_pct if cutoff_pct is not None else 100) / 100)

    if step is None:
        cond_indices = list(range(cutoff))
        label = f"limited_{cutoff_pct}pct"
        desc  = f"Limited view: first {cutoff_pct}% = {len(cond_indices)} of {total_angles} angles"
    elif cutoff_pct is None:
        cond_indices = [i for i in range(total_angles) if i % step == 0 and i < cutoff]
        label = f"sparse_step{step}"
        desc  = f"Sparse: every {step}th angle within first 75% = {len(cond_indices)} of {total_angles} angles"
    else:
        cond_indices = [i for i in range(total_angles) if i % step == 0 and i < cutoff]
        label = f"sparse_step{step}_cutoff{cutoff_pct}pct"
        desc  = f"Sparse+limited: every {step}th within first {cutoff_pct}% = {len(cond_indices)} of {total_angles} angles"

    n_known = len(cond_indices)
    print(f"\n{'='*80}")
    print(desc)
    print(f"{'='*80}")

    save_dir = f"Comparsion/FBP_TV_{label}_{n_known}of{total_angles}"
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

            print(f"  Slice {len(all_metrics)-1}: FBP+TV SSIM={metrics['tv_ssim']:.4f} PSNR={metrics['tv_psnr']:.2f}")

    if all_metrics:
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print(f"\n--- {label} average over {len(all_metrics)} slices ---")
        print(f"  FBP+TV: SSIM={avg['tv_ssim']:.4f}  PSNR={avg['tv_psnr']:.2f}")

        csv_path = os.path.join(save_dir, 'metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['slice'] + list(all_metrics[0].keys()))
            writer.writeheader()
            for i, m in enumerate(all_metrics):
                writer.writerow({'slice': i, **m})

        summary_path = os.path.join(save_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Summary — {label} ({len(all_metrics)} slices)\n")
            f.write("=" * 50 + "\n")
            f.write(f"FBP+TV  SSIM={avg['tv_ssim']:.4f}  PSNR={avg['tv_psnr']:.2f}\n")

    return avg if all_metrics else None


def main():
    data_path = "data/test_data"
    detector_count = 816
    angle_step = 360 / 720

    total_series = load_series_from(data_path)

    # ── Edit these two lists to configure runs ────────────────────────────────
    # Each index is one run. Use None to leave that dimension unset.
    #   cutoff_pct only  → contiguous limited-view
    #   step only        → sparse every-Nth within 75%
    #   both             → sparse every-Nth within cutoff_pct%
    cutoff_pcts  = [6.25, 12.5, 25, 50, 75, 100, 100, 100, 100, 100, 100, 6.25, 6.25, 12.5, 12.5, 25, 25,50, 50, 75, 75]
    sparse_steps = [None, None, None, None, None, None, 1, 2, 4, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10, 8, 10]  # e.g. [10,  None, 10]
    # ─────────────────────────────────────────────────────────────────────────

    assert len(cutoff_pcts) == len(sparse_steps), "Lists must be the same length"

    summary = {}

    for i, (pct, step) in enumerate(zip(cutoff_pcts, sparse_steps)):
        avg = run_setting(detector_count, angle_step, total_series,
                          cutoff_pct=pct, step=step)
        if avg:
            summary[i] = (pct, step, avg)

    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON — FBP+TV")
    print(f"{'='*80}")
    print(f"{'Run':<4} {'Setting':<30} {'Angles':<8} {'SSIM':<10} {'PSNR':<10}")
    print("-" * 64)
    for i, (pct, step, avg) in summary.items():
        cutoff = int(720 * (pct if pct is not None else 100) / 100)
        if step is None:
            n = cutoff
            tag = f"limited {pct}%"
        elif pct is None:
            n = len([j for j in range(720) if j % step == 0 and j < cutoff])
            tag = f"sparse step={step}"
        else:
            n = len([j for j in range(720) if j % step == 0 and j < cutoff])
            tag = f"step={step} cutoff={pct}%"
        print(f"{i:<4} {tag:<30} {n:<8} {avg['tv_ssim']:<10.4f} {avg['tv_psnr']:<10.2f}")


if __name__ == '__main__':
    main()
