# Baselines for SD-Flow

Six baselines, each in its own folder, trained and tested on exactly SD-Flow's
data, measured views and label:

| Folder | Method | Trained? | Free parameter |
|---|---|---|---|
| `fbp/` | FBP of the measured views | no | none |
| `sart/` | SART (ASTRA), non-negative | no | sweeps, picked on val |
| `tv/` | TV-regularized reconstruction (FISTA) | no | TV weight, picked on val |
| `fbpconvnet/` | FBPConvNet (Jin et al. 2017) | yes | best val checkpoint |
| `dolce/` | DOLCE (Liu et al., ICCV 2023) | yes | prox variant and weight, picked on val |
| `sword/` | SWORD (Xu et al., TMI 2024) | yes (2 models) | none |

## What is shared, so the comparison is fair

* **Data split.** The patient folders under `data/` in natural sort order: the
  first 7 train, the next 1 validation, the last 2 test. Low-dose series are
  skipped. `prepare_data.py` writes the split to
  `data/baselines_cache/<setting>/split.json`; check it before trusting results.
* **Input.** Sinograms come from SD-Flow's own `dicom_dataset`
  (`data/dicom_preprocess.py`): 720 views x 816 detectors, each normalized to
  [-1, 1]. Every method sees only the measured rows SD-Flow conditions on
  (`cond_indices`, same arithmetic as `main_diff_bfs.py`). The default is
  0-45 deg with every 10th row, i.e. 9 of 720 views. FBPConvNet's input is the FBP
  of those rows, the same image SD-Flow's third conditioning channel is built from.
* **Label and metrics.** The label is the FBP of the full sinogram
  (`test_diff._fbp_reconstruct`), the image SD-Flow is scored against. SSIM and
  PSNR follow `validation/compare_ssim.py`: normalize by the label's min/max,
  use data_range 1.
* **Pairing.** `evaluate.py` matches SD-Flow outputs to test slices by their
  ground-truth sinogram, not by folder order. It refuses a run made with
  different measured rows.
* **Tuning.** Every free parameter is chosen on the validation patient, never on test.

## Running (GAIVI)

Every step is its own single-GPU job. Submit it from the repo root; logs go to
`slurm-bl-<step>-<jobid>.out`. Run `prepare.sh` first; after that, the jobs
depend only on what's listed in the "After" column.

| Job | Command | After | Time |
|---|---|---|---|
| Data | `sbatch baselines/prepare.sh` | nothing | ~1-2 h |
| FBP | `sbatch baselines/fbp/run.sh` | prepare | minutes |
| SART | `sbatch baselines/sart/run.sh` | prepare | < 1 h |
| TV | `sbatch baselines/tv/run.sh` | prepare | ~1 h |
| FBPConvNet | `sbatch baselines/fbpconvnet/run.sh` | prepare | ~12 h (train + test) |
| DOLCE train | `sbatch baselines/dolce/train.sh` | prepare | 22 h |
| DOLCE sample | `sbatch baselines/dolce/sample.sh` | DOLCE train | ~4 h |
| SWORD train | `sbatch baselines/sword/train.sh full` and `... high` | prepare | 22 h each, in parallel |
| SWORD sample | `sbatch baselines/sword/sample.sh` | both SWORD trains | ~5 h |
| Table + figures | `sbatch baselines/evaluate.sh` | whatever is done | minutes |

Details:
* Pass extra arguments after the script, e.g. `sbatch baselines/tv/run.sh --tune_slices 16`.
* Set the patient folder with `DATA_ROOT` (default `data`) and training time with
  `TRAIN_HOURS` (FBPConvNet 12, DOLCE and SWORD 22).
* A job that stops early can simply be resubmitted: training resumes from
  `last.pt` and sampling skips slices already done. Longer training
  (`TRAIN_HOURS` above 22.5) needs one resubmission per extra day.
* `baselines/run_all.sh` submits all of the above at once, with the dependencies.
* On a GPU machine without SLURM, run
  `python -m baselines.<folder>.<script> --help` directly.

## Adding SD-Flow to the table

Evaluate SD-Flow on each test series with the same angles (`angle_start 0`,
`angle_end 45`, `angle_stride 10` in the config JSON, `data_path` pointing at
the series folder), then:

```bash
python -m baselines.evaluate \
  --sdflow "SD-Flow=best_model_folder/<run>/diffusion_folder/<series1>/contour,best_model_folder/<run>/diffusion_folder/<series2>/contour" \
  --sdflow "Physics only=..."
```

Results are written to `output/baselines/<setting>/evaluation/`:
- `summary.md` / `summary.csv`: mean ± std over the slices every method has
- `per_slice.csv`
- `figures/`: one shared display window for every panel

## Implementation notes and deviations

* **Units.** The iterative methods fit y = s + 1 (the normalized sinogram is a
  rescaled projection whose minimum, air, is 0). Because FBP is linear, their
  images map to the label's units by subtracting K = FBP(1) (`common/ct.py`).
* **TV** solves the regularized inverse problem. The old `dicom_fbp_degraded.py`
  "TV" was FBP followed by TV denoising, which is why it looked like FBP.
* **DOLCE** follows the paper: condition on an RLS image, 20% condition dropout,
  T = 2000 linear, Adam 1.5e-4, dropout 0.2, EMA, latest checkpoint, 40-step
  DDIM (eta 1), and a CG prox for data consistency after every step. Differences:
  - batch 8 instead of 256, and a smaller U-Net (base 64) so it trains in about a day
  - no learned variance, which DDIM sampling doesn't use
  - RLS is ||Az - y||² + beta ||grad z||², solved with CG. The paper doesn't define
    its RLS beyond "regularized least squares".

  `--tune` compares the paper's prox-on-sample with prox-on-x0 and three prox
  weights on validation slices.
* **SWORD** follows the official loop:
  - Haar wavelet bands of the sinogram, two VE-SDE score models (full-frequency
    and high-frequency)
  - sampling starts at level 550 of 2000, from the zero-filled sinogram
  - reverse-diffusion predictor plus Langevin corrector (snr 0.16)

  Differences:
  - sigma_max is estimated from the data (Song's rule, which is where SWORD's
    378 comes from). Both models share it, as SWORD's two configs share 378.
  - NCSN++ is replaced by the shared U-Net with Fourier embeddings of log sigma
  - sampling is standard stochastic PC: noise at the start and in every update,
    and no corrector at the final, noise-free level. The official loop keeps each
    update's mean. In an exact-oracle test that drifts to 30x the noise level
    mid-schedule, while the stochastic sampler tracks it exactly.
    `--official_mean` restores the official behavior.
  - data consistency only writes back measured sinogram rows. The official code
    also copies ground-truth wavelet coefficients, which leaks unmeasured rows.
* **FBPConvNet**: the original U-Net with a residual connection. Adam 1e-4 with
  cosine decay replaces the paper's SGD.

## Local smoke test

ASTRA has no CPU fan-beam FBP, so without a GPU `common/ct.py` falls back to an
approximate FBP and warns. That's enough to exercise the code, but never report
numbers from it.
