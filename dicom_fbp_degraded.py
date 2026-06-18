import argparse
import glob
import os

import astra
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from skimage.restoration import denoise_tv_chambolle


# ── DICOM & physics helpers ──────────────────────────────────────────────────

def load_dicom_hu_and_spacing(dicom_path):
    dcm = pydicom.dcmread(dicom_path)
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    hu = dcm.pixel_array.astype(np.float32) * slope + intercept
    spacing = getattr(dcm, "PixelSpacing", [1.0, 1.0])
    return hu, float(spacing[1]), float(spacing[0])  # dx, dy


def hu_to_mu(hu):
    return np.clip((0.02 * (1.0 + hu / 1000.0)).astype(np.float32), 0.0, None)


def mu_to_hu(mu):
    mu = np.clip(mu.astype(np.float32), 0.0, None)
    return np.clip(1000.0 * (mu / 0.02 - 1.0), -1024.0, 3071.0)


# ── ASTRA forward / inverse ─────────────────────────────────────────────────

def create_sinogram(hu, dx, dy, det_count, num_angles, dso, odd):
    mu = hu_to_mu(hu)
    h, w = mu.shape
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False).astype(np.float32)

    vol_geom = astra.create_vol_geom(h, w, -w*dx/2, w*dx/2, -h*dy/2, h*dy/2)
    proj_geom = astra.create_proj_geom("fanflat", dx, det_count, angles, dso, odd)
    projector_id = astra.create_projector("line_fanflat", proj_geom, vol_geom)

    vol_id = astra.data2d.create("-vol", vol_geom, np.ascontiguousarray(mu))
    sino_id, sino = astra.create_sino(vol_id, projector_id)

    astra.data2d.delete(sino_id)
    astra.data2d.delete(vol_id)
    astra.projector.delete(projector_id)
    return sino.astype(np.float32), angles


def fbp_reconstruct(sino, angles, det_spacing, dso, odd, n_pix):
    sino = np.ascontiguousarray(sino, dtype=np.float32)
    _, det_count = sino.shape

    vol_geom = astra.create_vol_geom(n_pix, n_pix)
    proj_geom = astra.create_proj_geom("fanflat", det_spacing, det_count, angles, dso, odd)

    sino_id = astra.data2d.create("-sino", proj_geom, sino)
    rec_id = astra.data2d.create("-vol", vol_geom)

    cfg = astra.astra_dict("FBP_CUDA")
    cfg["ProjectionDataId"] = sino_id
    cfg["ReconstructionDataId"] = rec_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    rec = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)
    return rec.astype(np.float32)


def sart_reconstruct(sino, angles, det_spacing, dso, odd, n_pix, iterations=200):
    sino = np.ascontiguousarray(sino, dtype=np.float32)
    _, det_count = sino.shape

    vol_geom = astra.create_vol_geom(n_pix, n_pix)
    proj_geom = astra.create_proj_geom("fanflat", det_spacing, det_count, angles, dso, odd)

    sino_id = astra.data2d.create("-sino", proj_geom, sino)
    rec_id = astra.data2d.create("-vol", vol_geom)

    cfg = astra.astra_dict("SART_CUDA")
    cfg["ProjectionDataId"] = sino_id
    cfg["ReconstructionDataId"] = rec_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iterations)
    rec = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)
    return rec.astype(np.float32)


def tv_reconstruct(sino, angles, det_spacing, dso, odd, n_pix, tv_weight=0.002):
    """FBP followed by total-variation denoising (Chambolle)."""
    rec_mu = fbp_reconstruct(sino, angles, det_spacing, dso, odd, n_pix)
    r_min, r_max = float(rec_mu.min()), float(rec_mu.max())
    if r_max > r_min:
        normed = (rec_mu - r_min) / (r_max - r_min)
        denoised = denoise_tv_chambolle(normed, weight=tv_weight)
        rec_mu = (denoised * (r_max - r_min) + r_min).astype(np.float32)
    return rec_mu


# ── Angle selection ──────────────────────────────────────────────────────────

def make_limited_sparse_indices(num_angles, limit_deg, index_step):
    """Select every `index_step`-th angle within the first `limit_deg` degrees.

    With 720 angles over 360°, each index = 0.5°.
    limit_deg=45 → first 90 indices, index_step=10 → indices 0,10,20,...,80
    → angles at 0°, 5°, 10°, ..., 40°  (9 views).
    """
    cutoff = int(round(num_angles * limit_deg / 360.0))
    cutoff = min(cutoff, num_angles)
    indices = np.arange(0, cutoff, index_step, dtype=np.int64)
    return indices


# ── Plotting helpers ─────────────────────────────────────────────────────────

def save_panel_pdf(img, title, out_path, *, cmap="gray", vmin=None, vmax=None,
                   aspect="auto", colorbar_label=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(img, cmap=cmap, aspect=aspect)
    if vmin is not None:
        im.set_clim(vmin=vmin)
    if vmax is not None:
        im.set_clim(vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    if colorbar_label is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── Shared save routine for one method ───────────────────────────────────────

def save_method_outputs(stem, out_dir, method_label, mode_title,
                        sino_full, sino_masked, full_rec_hu, method_rec_hu,
                        keep_idx, vmin, vmax, s_lo, s_hi, window_width, window_level):
    os.makedirs(out_dir, exist_ok=True)

    sino_diff = np.abs(sino_full - sino_masked)
    recon_diff = np.abs(full_rec_hu - method_rec_hu)

    # ── 2×3 comparison panel ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(sino_full, cmap="gray", aspect="auto", vmin=s_lo, vmax=s_hi)
    axes[0, 0].set_title("Sinogram Full")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(sino_masked, cmap="gray", aspect="auto", vmin=s_lo, vmax=s_hi)
    axes[0, 1].set_title(f"Sinogram Masked – {mode_title}")
    axes[0, 1].axis("off")

    im_sd = axes[0, 2].imshow(sino_diff, cmap="hot", aspect="auto")
    axes[0, 2].set_title("|Sinogram Difference|")
    axes[0, 2].axis("off")
    fig.colorbar(im_sd, ax=axes[0, 2], fraction=0.046, pad=0.04).set_label("Absolute Error")

    axes[1, 0].imshow(full_rec_hu, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Reconstruction – Full Sinogram")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(method_rec_hu, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f"{method_label} – {mode_title}")
    axes[1, 1].axis("off")

    im_rd = axes[1, 2].imshow(recon_diff, cmap="hot")
    axes[1, 2].set_title("|Reconstruction Difference| (HU)")
    axes[1, 2].axis("off")
    fig.colorbar(im_rd, ax=axes[1, 2], fraction=0.046, pad=0.04).set_label("Absolute Error (HU)")

    fig.suptitle(
        f"{method_label} Comparison (W={window_width}, L={window_level})\n"
        f"dicom={stem} | {mode_title}",
        fontsize=12,
    )
    plt.tight_layout()

    png_path = os.path.join(out_dir, f"{stem}_comparison.png")
    pdf_path = os.path.join(out_dir, f"{stem}_comparison.pdf")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # ── individual panel PDFs ────────────────────────────────────────────
    panels = {
        "sino_full": (sino_full, "Sinogram Full",
                      dict(cmap="gray", vmin=s_lo, vmax=s_hi, aspect="auto")),
        "sino_masked": (sino_masked, f"Sinogram Masked – {mode_title}",
                        dict(cmap="gray", vmin=s_lo, vmax=s_hi, aspect="auto")),
        "sino_diff": (sino_diff, "|Sinogram Difference|",
                      dict(cmap="hot", aspect="auto", colorbar_label="Absolute Error")),
        "recon_full": (full_rec_hu, "Reconstruction – Full Sinogram",
                       dict(cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")),
        "recon_method": (method_rec_hu, f"{method_label} – {mode_title}",
                         dict(cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")),
        "recon_diff": (recon_diff, "|Reconstruction Difference| (HU)",
                       dict(cmap="hot", aspect="equal", colorbar_label="Absolute Error (HU)")),
    }

    for key, (img, title, kwargs) in panels.items():
        save_panel_pdf(img, title, os.path.join(out_dir, f"{stem}_panel_{key}.pdf"), **kwargs)

    # ── standalone reconstruction ────────────────────────────────────────
    single_path = os.path.join(out_dir, f"{stem}_reconstruction_only.pdf")
    plt.figure(figsize=(8, 8))
    plt.imshow(method_rec_hu, cmap="gray", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(single_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # ── .npy arrays ──────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, f"{stem}_sinogram_full.npy"), sino_full)
    np.save(os.path.join(out_dir, f"{stem}_sinogram_masked.npy"), sino_masked)
    np.save(os.path.join(out_dir, f"{stem}_keep_indices.npy"), keep_idx)
    np.save(os.path.join(out_dir, f"{stem}_recon_full_hu.npy"), full_rec_hu)
    np.save(os.path.join(out_dir, f"{stem}_recon_method_hu.npy"), method_rec_hu)

    print(f"    [{method_label}] → {out_dir}")


# ── Per-DICOM processing ────────────────────────────────────────────────────

def process_dicom(dicom_path, output_root, *, limit_deg, index_step,
                  det_count, num_angles, dso, odd, n_pix,
                  window_level, window_width,
                  sart_iterations, tv_weight):

    stem = os.path.splitext(os.path.basename(dicom_path))[0]
    hu, dx, dy = load_dicom_hu_and_spacing(dicom_path)

    # Full sinogram & ground-truth reconstruction
    sino_full, angles_full = create_sinogram(
        hu, dx, dy, det_count, num_angles, dso, odd,
    )
    full_rec_hu = mu_to_hu(
        fbp_reconstruct(sino_full, angles_full, dx, dso, odd, n_pix)
    )

    # Degraded sinogram
    keep_idx = make_limited_sparse_indices(num_angles, limit_deg, index_step)
    sino_masked = np.zeros_like(sino_full)
    sino_masked[keep_idx, :] = sino_full[keep_idx, :]

    # Display ranges (shared across methods)
    vmin = window_level - window_width / 2.0
    vmax = window_level + window_width / 2.0
    s_lo = min(np.percentile(sino_full, 1), np.percentile(sino_masked, 1))
    s_hi = max(np.percentile(sino_full, 99), np.percentile(sino_masked, 99))

    deg_per_idx = 360.0 / num_angles
    step_deg = deg_per_idx * index_step
    mode_title = f"Limited 0-{limit_deg:g}°, step {step_deg:g}° ({len(keep_idx)} views)"
    folder_tag = f"limited{limit_deg:g}deg_step{step_deg:g}deg_{len(keep_idx)}views"

    shared = dict(
        stem=stem,
        mode_title=mode_title,
        sino_full=sino_full,
        sino_masked=sino_masked,
        full_rec_hu=full_rec_hu,
        keep_idx=keep_idx,
        vmin=vmin, vmax=vmax,
        s_lo=s_lo, s_hi=s_hi,
        window_width=window_width,
        window_level=window_level,
    )

    # ── FBP ──────────────────────────────────────────────────────────────
    fbp_rec_hu = mu_to_hu(
        fbp_reconstruct(sino_masked, angles_full, dx, dso, odd, n_pix)
    )
    save_method_outputs(
        out_dir=os.path.join(output_root, f"FBP_{folder_tag}"),
        method_label="FBP",
        method_rec_hu=fbp_rec_hu,
        **shared,
    )

    # ── SART ─────────────────────────────────────────────────────────────
    sart_rec_hu = mu_to_hu(
        sart_reconstruct(sino_masked, angles_full, dx, dso, odd, n_pix,
                         iterations=sart_iterations)
    )
    save_method_outputs(
        out_dir=os.path.join(output_root, f"SART_{folder_tag}"),
        method_label=f"SART ({sart_iterations} iter)",
        method_rec_hu=sart_rec_hu,
        **shared,
    )

    # ── TV (FBP + Chambolle TV denoising) ────────────────────────────────
    tv_rec_hu = mu_to_hu(
        tv_reconstruct(sino_masked, angles_full, dx, dso, odd, n_pix,
                       tv_weight=tv_weight)
    )
    save_method_outputs(
        out_dir=os.path.join(output_root, f"TV_{folder_tag}"),
        method_label=f"FBP+TV (w={tv_weight})",
        method_rec_hu=tv_rec_hu,
        **shared,
    )

    print(f"  [{stem}] Done — {len(keep_idx)} views kept")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch CT reconstruction comparison: FBP, SART, and FBP+TV."
    )
    parser.add_argument("--dicom_folder", required=True, help="Folder with DICOM .dcm files.")
    parser.add_argument("--output_root", default="Comparison", help="Root output directory.")
    parser.add_argument("--limit_deg", type=float, default=45.0,
                        help="Angular range in degrees (default: 45).")
    parser.add_argument("--index_step", type=int, default=10,
                        help="Take every N-th index within the limited range (default: 10).")
    parser.add_argument("--det_count", type=int, default=736)
    parser.add_argument("--num_angles", type=int, default=720)
    parser.add_argument("--dso", type=float, default=1000.0)
    parser.add_argument("--odd", type=float, default=600.0)
    parser.add_argument("--n_pix", type=int, default=512)
    parser.add_argument("--window_level", type=float, default=40.0)
    parser.add_argument("--window_width", type=float, default=350.0)
    parser.add_argument("--sart_iterations", type=int, default=200,
                        help="Number of SART iterations (default: 200).")
    parser.add_argument("--tv_weight", type=float, default=0.002,
                        help="TV denoising weight for Chambolle (default: 0.002).")
    parser.add_argument("--glob_pattern", default="*.dcm", help="DICOM filename pattern.")
    args = parser.parse_args()

    dicom_files = sorted(glob.glob(os.path.join(args.dicom_folder, args.glob_pattern)))
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {args.dicom_folder}")

    os.makedirs(args.output_root, exist_ok=True)
    print(f"Found {len(dicom_files)} DICOM files → {args.output_root}")
    print(f"Methods: FBP | SART ({args.sart_iterations} iter) | FBP+TV (w={args.tv_weight})")

    for i, path in enumerate(dicom_files, 1):
        print(f"[{i}/{len(dicom_files)}] {path}")
        process_dicom(
            path, args.output_root,
            limit_deg=args.limit_deg,
            index_step=args.index_step,
            det_count=args.det_count,
            num_angles=args.num_angles,
            dso=args.dso,
            odd=args.odd,
            n_pix=args.n_pix,
            window_level=args.window_level,
            window_width=args.window_width,
            sart_iterations=args.sart_iterations,
            tv_weight=args.tv_weight,
        )

    print("Done.")


if __name__ == "__main__":
    main()