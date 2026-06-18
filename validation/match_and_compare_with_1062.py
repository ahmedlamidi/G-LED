import argparse
import os

import astra
import numpy as np
import pydicom
from matplotlib import pyplot as plt


def prepare_sinogram_2d(arr):
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        axis = int(np.argmin(arr.shape))
        idx = arr.shape[axis] // 2
        arr = np.take(arr, idx, axis=axis)
        if arr.ndim == 2:
            return arr.astype(np.float32)
    raise ValueError(f"Expected 2D sinogram, got shape {arr.shape}")


def load_dicom_hu_and_spacing(dicom_path):
    dcm = pydicom.dcmread(dicom_path)
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))

    pixels = dcm.pixel_array.astype(np.float32)
    hu = pixels * slope + intercept

    spacing = getattr(dcm, "PixelSpacing", [1.0, 1.0])
    dy = float(spacing[0])
    dx = float(spacing[1])
    return hu, dx, dy


def hu_to_mu(hu):
    mu = 0.02 * (1.0 + hu / 1000.0)
    return np.clip(mu.astype(np.float32), 0.0, None)


def mu_to_hu(mu):
    mu = np.clip(mu.astype(np.float32), 0.0, None)
    hu = 1000.0 * (mu / 0.02 - 1.0)
    return np.clip(hu, -1024.0, 3071.0)


def create_sinogram_from_hu(hu, dx, dy, det_count, num_angles, dso, odd):
    mu = hu_to_mu(hu)
    h, w = mu.shape
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False).astype(np.float32)

    vol_geom = astra.create_vol_geom(
        h,
        w,
        -w * dx / 2.0,
        w * dx / 2.0,
        -h * dy / 2.0,
        h * dy / 2.0,
    )
    proj_geom = astra.create_proj_geom("fanflat", dx, det_count, angles, dso, odd)
    projector_id = astra.create_projector("line_fanflat", proj_geom, vol_geom)

    vol_id = astra.data2d.create("-vol", vol_geom, np.ascontiguousarray(mu, dtype=np.float32))
    sino_id, sino = astra.create_sino(vol_id, projector_id)

    astra.data2d.delete(sino_id)
    astra.data2d.delete(vol_id)
    astra.projector.delete(projector_id)
    return sino.astype(np.float32)


def fbp_reconstruct(sino, dso, odd, det_spacing, n_pix):
    sino = np.ascontiguousarray(sino, dtype=np.float32)
    num_angles, det_count = sino.shape
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False).astype(np.float32)

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


def match_sinogram_range(source, reference):
    s_min, s_max = float(np.min(source)), float(np.max(source))
    r_min, r_max = float(np.min(reference)), float(np.max(reference))
    if s_max <= s_min:
        return np.full_like(source, (r_min + r_max) * 0.5, dtype=np.float32)

    out = (source - s_min) / (s_max - s_min)
    out = out * (r_max - r_min) + r_min
    return out.astype(np.float32)


def save_panel_pdf(img, title, out_path, *, cmap="gray", vmin=None, vmax=None, aspect="auto", colorbar_label=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if aspect == "equal":
        im = ax.imshow(img, cmap=cmap, aspect="equal")
    else:
        im = ax.imshow(img, cmap=cmap, aspect="auto")

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


def main():
    parser = argparse.ArgumentParser(description="Create sinogram from 1-062.dcm, range-match model sinogram, reconstruct both, and plot.")
    parser.add_argument("--model_sinogram", required=True, help="Path to model sinogram .npy (e.g., recon_micro_0.npy).")
    parser.add_argument("--dicom_path", default="data/extra_data/1-062.dcm", help="Reference DICOM file path.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. If omitted, uses <model_sinogram_folder>/sino_match_compare.",
    )
    parser.add_argument("--dso", type=float, default=1000.0)
    parser.add_argument("--odd", type=float, default=600.0)
    parser.add_argument("--n_pix", type=int, default=512)
    parser.add_argument("--window_level", type=float, default=40.0)
    parser.add_argument("--window_width", type=float, default=350.0)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.model_sinogram), "sino_match_compare")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Resolved output_dir: {args.output_dir}")

    model_sino = prepare_sinogram_2d(np.load(args.model_sinogram))
    num_angles, det_count = model_sino.shape

    ref_hu, dx, dy = load_dicom_hu_and_spacing(args.dicom_path)
    ref_sino = create_sinogram_from_hu(
        ref_hu,
        dx=dx,
        dy=dy,
        det_count=det_count,
        num_angles=num_angles,
        dso=args.dso,
        odd=args.odd,
    )

    matched_model_sino = match_sinogram_range(model_sino, ref_sino)

    ref_rec_mu = fbp_reconstruct(ref_sino, dso=args.dso, odd=args.odd, det_spacing=dx, n_pix=args.n_pix)
    model_rec_mu = fbp_reconstruct(matched_model_sino, dso=args.dso, odd=args.odd, det_spacing=dx, n_pix=args.n_pix)

    ref_rec_hu = mu_to_hu(ref_rec_mu)
    model_rec_hu = mu_to_hu(model_rec_mu)

    vmin = args.window_level - args.window_width / 2.0
    vmax = args.window_level + args.window_width / 2.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    s_lo = min(np.percentile(ref_sino, 1), np.percentile(matched_model_sino, 1))
    s_hi = max(np.percentile(ref_sino, 99), np.percentile(matched_model_sino, 99))

    axes[0, 0].imshow(ref_sino, cmap="gray", aspect="auto", vmin=s_lo, vmax=s_hi)
    axes[0, 0].set_title("Sinogram from 1-062 DICOM")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(matched_model_sino, cmap="gray", aspect="auto", vmin=s_lo, vmax=s_hi)
    axes[0, 1].set_title("Model Sinogram (Range-matched)")
    axes[0, 1].axis("off")

    sino_diff = np.abs(ref_sino - matched_model_sino)
    recon_diff = np.abs(ref_rec_hu - model_rec_hu)

    im_sino_diff = axes[0, 2].imshow(sino_diff, cmap="hot", aspect="auto")
    axes[0, 2].set_title("|Sinogram Difference|")
    axes[0, 2].axis("off")
    cbar_sino = fig.colorbar(im_sino_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar_sino.set_label("Absolute Error")

    axes[1, 0].imshow(ref_rec_hu, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Reconstruction from 1-062 Sinogram")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(model_rec_hu, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Reconstruction from Matched Model Sinogram")
    axes[1, 1].axis("off")

    im_recon_diff = axes[1, 2].imshow(recon_diff, cmap="hot")
    axes[1, 2].set_title("|Reconstruction Difference| (HU)")
    axes[1, 2].axis("off")
    cbar_recon = fig.colorbar(im_recon_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar_recon.set_label("Absolute Error (HU)")

    fig.suptitle(
        f"Comparison (W={args.window_width}, L={args.window_level})\n"
        f"model={os.path.basename(args.model_sinogram)} | ref={os.path.basename(args.dicom_path)}",
        fontsize=12,
    )
    plt.tight_layout()

    png_path = os.path.join(args.output_dir, "sino_recon_comparison.png")
    pdf_path = os.path.join(args.output_dir, "sino_recon_comparison.pdf")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # Save each panel as an individual PDF
    panel_paths = {
        "sino_reference": os.path.join(args.output_dir, "panel_sino_reference.pdf"),
        "sino_model_matched": os.path.join(args.output_dir, "panel_sino_model_matched.pdf"),
        "sino_difference": os.path.join(args.output_dir, "panel_sino_difference.pdf"),
        "recon_reference": os.path.join(args.output_dir, "panel_recon_reference.pdf"),
        "recon_model_matched": os.path.join(args.output_dir, "panel_recon_model_matched.pdf"),
        "recon_difference": os.path.join(args.output_dir, "panel_recon_difference.pdf"),
    }

    save_panel_pdf(
        ref_sino,
        "Sinogram from 1-062 DICOM",
        panel_paths["sino_reference"],
        cmap="gray",
        vmin=s_lo,
        vmax=s_hi,
        aspect="auto",
    )
    save_panel_pdf(
        matched_model_sino,
        "Model Sinogram (Range-matched)",
        panel_paths["sino_model_matched"],
        cmap="gray",
        vmin=s_lo,
        vmax=s_hi,
        aspect="auto",
    )
    save_panel_pdf(
        sino_diff,
        "|Sinogram Difference|",
        panel_paths["sino_difference"],
        cmap="hot",
        aspect="auto",
        colorbar_label="Absolute Error",
    )
    save_panel_pdf(
        ref_rec_hu,
        "Reconstruction from 1-062 Sinogram",
        panel_paths["recon_reference"],
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    save_panel_pdf(
        model_rec_hu,
        "Reconstruction from Matched Model Sinogram",
        panel_paths["recon_model_matched"],
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    save_panel_pdf(
        recon_diff,
        "|Reconstruction Difference| (HU)",
        panel_paths["recon_difference"],
        cmap="hot",
        aspect="equal",
        colorbar_label="Absolute Error (HU)",
    )

    # Save matched-model reconstruction as standalone PDF
    single_pdf_path = os.path.join(args.output_dir, "reconstruction_matched_model_only.pdf")
    plt.figure(figsize=(8, 8))
    plt.imshow(model_rec_hu, cmap="gray", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(single_pdf_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    np.save(os.path.join(args.output_dir, "reference_sinogram_1062.npy"), ref_sino)
    np.save(os.path.join(args.output_dir, "model_sinogram_matched.npy"), matched_model_sino)
    np.save(os.path.join(args.output_dir, "recon_reference_hu.npy"), ref_rec_hu)
    np.save(os.path.join(args.output_dir, "recon_model_hu.npy"), model_rec_hu)

    print(f"Saved comparison plot: {png_path}")
    print(f"Saved comparison plot: {pdf_path}")
    print(f"Saved standalone matched-model reconstruction: {single_pdf_path}")
    print("Saved panel PDFs:")
    for key, path in panel_paths.items():
        print(f"  {key}: {path}")
    print(f"Saved arrays in: {args.output_dir}")


if __name__ == "__main__":
    main()
