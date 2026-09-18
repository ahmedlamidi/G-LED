# Baselines for SD-Flow

Eight baselines, each in its own folder, trained and tested on exactly SD-Flow's
data, measured views and label:

| Folder | Method | Trained? | Free parameter |
|---|---|---|---|
| `fbp/` | FBP of the measured views | no | none |
| `sart/` | SART (ASTRA), non-negative | no | sweeps, picked on val |
| `tv/` | TV-regularized reconstruction (FISTA) | no | TV weight, picked on val |
| `fbpconvnet/` | FBPConvNet (Jin et al. 2017) | yes | best val checkpoint |
| `dolce/` | DOLCE (Liu et al., ICCV 2023) | yes | prox variant and weight, picked on val |
| `sword/` | SWORD (Xu et al., TMI 2024) | yes (2 models) | none |
| `dps/` | DPS (Chung et al., ICLR 2023) | one unconditional prior for all settings (`dps/train_prior.sh`) | step size zeta, picked on val |
| `dudotrans/` | DuDoTrans (Wang et al., 2021) | yes | best val checkpoint |

## What is shared, so the comparison is fair

* **Data split.** LIDC-IDRI-0001 to 0010 from `data/LIDC-IDRI` (the first 10
  patient folders in natural order): 0001-0007 train, 0008 validation, 0009-0010
  test. Only CT series are read. The chest X-rays (DX/CR) and nodule
  segmentations (SEG) stored with each patient are skipped.
  `prepare_data.py` writes the split to
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
  ground-truth sinogram, not by folder order. SD-Flow may use other measured
  views than the baselines; `summary.md` then lists each run's views.
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
| DPS prior | `sbatch baselines/dps/train_prior.sh` | prepare | ~17-22 h, once |
| DPS sample | `sbatch baselines/dps/sample.sh` | DPS prior (else DOLCE's checkpoint) | ~30 min tuning + ~1 min per slice |
| DuDoTrans | `sbatch baselines/dudotrans/run.sh` | prepare | up to 20 h (validation patience), test in minutes |
| SD-Flow sample | `sbatch baselines/sdflow/sample.sh --config configurations/<run>.json` | prepare, a trained SD-Flow | depends on its sampling steps |
| View sweep | `sbatch baselines/sweep.sh` | both SWORD trains | ~1 h (SWORD on 8 settings x 10 slices) |
| Per-setting training | `METHOD=dolce sbatch baselines/train_sweep.sh` | prepare | FBPConvNet ~1 day, DOLCE several days, one job at a time |
| Table + figures | `sbatch baselines/evaluate.sh` | whatever is done | minutes |

Details:
* Pass extra arguments after the script, e.g. `sbatch baselines/tv/run.sh --tune_slices 16`.
* `prepare.sh` for a second view setting can share the label with the first
  (`--label_from data/baselines_cache/limited0-45_stride10`) and write only the
  arrays a method needs (`--images label,rls` for DOLCE, `label,fbp_in` for
  FBPConvNet); arrays already present are kept. Each array is ~1 MB per slice.
* Set the patient folder with `DATA_ROOT` (default `data/LIDC-IDRI`) and training time with
  `TRAIN_HOURS` (FBPConvNet 12, DOLCE and SWORD 22).
* A job that stops early can simply be resubmitted: training resumes from
  `last.pt` and sampling skips slices already done. Longer training
  (`TRAIN_HOURS` above 22.5) needs one resubmission per extra day.
* `baselines/run_all.sh` submits all of the above at once, with the dependencies.
* On a GPU machine without SLURM, run
  `python -m baselines.<folder>.<script> --help` directly.

## The baselines over view settings

`baselines/sweep.py` scores the methods on 8 settings, arcs from 0 deg: 45, 90
and 270 deg every row; 360 deg every 2nd, 4th, 8th and 10th row; and the
baselines' 45 deg every 10th row. 10 evenly spaced test slices per setting,
the same ids everywhere; no RLS is built for the sweep's own folders.

* **FBP** is computed by the sweep itself.
* **SWORD** never sees which views are measured, so the pair trained once is
  sampled per setting by the sweep (`sbatch baselines/sweep.sh`, ~1 h). The
  reference setting reuses the full run's slices when present.
* **DPS** has an unconditional prior, so like SWORD it is sampled per setting by
  the sweep (`--methods fbp,sword,dps`); zeta is tuned per setting on the base
  setting's validation slices. About 50 min per setting.
* **DOLCE, FBPConvNet and DuDoTrans** learn the streaks of the measured views, so they are
  trained per setting with `METHOD=dolce sbatch baselines/train_sweep.sh`
  (likewise `METHOD=fbpconvnet`, `METHOD=dudotrans`): one job, settings one after another,
  resubmitting itself across the 24 h wall clock. DOLCE stops on its own when
  the mean training loss of the last 3 h improved on the 3 h before by less
  than 5% (`--plateau_hours`, `--plateau_tol`, `--min_hours 6`, cap
  `DOLCE_HOURS=22`); FBPConvNet stops after 15 epochs without a validation
  improvement (`--patience`). Then `sbatch baselines/sweep.sh --methods
  fbp,sword,dolce,fbpconvnet --no_sample` adds them to the table.

Results go to `output/baselines/sword_sweep/`: SWORD's per-setting
`<setting>/sword/{recon,sino}/`, and in `sweep/` the table (`summary.md`,
`summary.csv`, `per_slice.csv`: body, HU and legacy image metrics for every
method, SSIM of the completed sinogram for SWORD) `figures/grid_<method>.png` (the overview grid: slices x settings, label last) and
`figures/strip_<slice>.png`, one row per method with the image under every setting.
DOLCE's and FBPConvNet's per-setting results stay in `output/baselines/<setting>/`.

## Adding SD-Flow to the table

`baselines/sdflow/sample.py` runs SD-Flow on the same test slices as the
baselines, with the model folder, measured views and sampling steps of an
SD-Flow config (the conditioning and sampler of `main_diff_eval_bfs.py`), and
writes `output/baselines/<setting>/sdflow/recon/` like any other method:

```bash
sbatch baselines/sdflow/sample.sh --config configurations/<run>.json
sbatch baselines/sdflow/sample.sh --config configurations/<other>.json --name sdflow_other   # a second variant
sbatch baselines/evaluate.sh                                     # SD-Flow is in the default --methods
sbatch baselines/evaluate.sh --methods fbp,dolce,sdflow,sdflow_other
```

SD-Flow may use other views than the baselines; `summary.md` lists them.
Not sure which window a model was trained with? `baselines/sdflow/start_sweep.sh
--config <its config>` slides the window's start around the circle, scores each
start on validation slices and plots the curve; the trained window is the
peak, and it is also recorded in `<model>/diffusion_folder/logging/args.txt`.
Existing `main_diff_eval_bfs.py` outputs can still be imported with
`--sdflow "NAME=<diffusion_folder>/<series>/contour"`.

Results are written to `output/baselines/<setting>/evaluation/`:
- `summary.md` / `summary.csv`: mean ± std over the slices every method has
- `per_slice.csv`
- `figures/overview.png`: the standard overview grid, 10 evenly spaced slices (rows) x
  methods (columns) with the label last, body-cropped, one HU window, body SSIM per panel
- `figures/compare_<slice>.png`: a preview of every shared slice, all methods side
  by side in HU lung and soft-tissue windows, column order in `figures/columns.txt`
- `--export_slice <slice id>` instead writes `slice_<slice id>/<method>_<window>.pdf`,
  one image per method and window, for building the paper figure

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
* **DPS** follows Algorithm 1 of the paper: ancestral DDPM sampling, and at every
  step the gradient of ||y - A x0_hat||_2 (the norm, not its square) with respect
  to x_t, backpropagated through the denoiser. Differences:
  - the prior is DOLCE's network and training recipe with the condition always
    dropped (`dps/train_prior.sh` -> `<default setting>/dps_prior`, one model for all
    view settings, ~17-22 h), not the paper's ADM architecture. Until that training
    has finished, DOLCE's own checkpoint is used with its condition zeroed (20%
    condition dropout makes that branch an unconditional model, but with a fifth of
    the training). A folder holds results of one prior only (`--name` for another).
  - 1000 reverse steps as in the paper, by respacing DOLCE's T = 2000 schedule
    (every 2nd timestep) the way guided-diffusion does
  - A is the fan-beam projector in physical units (label + K), y = s + 1; its
    gradient comes from ASTRA's backprojection, which on the GPU is only
    approximately the adjoint of its projection
  - the gradient is divided by ||A|| and by the image normalization's scale so that
    zeta is comparable across view settings; zeta is then picked on 4 validation
    slices from {0.3, 1, 3, 10, 30} by SSIM inside the body. The paper sets zeta by
    hand per task and has no CT experiment.
  - the posterior mean uses x0_hat clipped to [-1, 1] (as the official code does);
    the residual uses the unclipped x0_hat
  - measurements are noise-free
* **DuDoTrans** follows the paper's structure, loss and optimizer: SRT with m = 3
  residual blocks of n = 1 Swin module + conv and Y~ = Y + residual; FBP consistency
  layer; RIRM with depth 2, width 4 and X~ = RIRM([X~1, X~2]) + X~1; loss L_SRT +
  L_DC + L_RIRM with both weights 1; Adam 1e-4, batch 1, up to 100 epochs. The
  official repository is an empty placeholder, so everything else is a choice:
  - Swin hyperparameters, which the paper does not state: dim 48, 4 heads, window 8,
    MLP ratio 4, one Swin module = a regular and a shifted-window block. 0.56 M
    parameters (paper: 0.44 M).
  - a stride-2 embedding conv and a pixel-shuffle reconstruction in both networks,
    so attention runs on 360 x 408 sinogram tokens and 256 x 256 image tokens. The
    sinogram here is 720 x 816, much larger than the paper's; `--patch 1` is full resolution.
  - the SRT works on the full 720-view sinogram: measured rows in place, rows
    between two measured rows one stride apart linearly interpolated, rows in larger
    gaps (outside a limited arc) zero, plus a mask channel. The paper only covers
    uniform sparse views and does not say how the missing views enter.
  - the measured rows are written back into the restored sinogram (`--no_data_consistency`
    turns that off); the paper does not mention it
  - the consistency layer is a direct fan-beam FBP in torch (`common/torch_ct.py`,
    exact gradient) instead of the paper's fan-to-parallel rebinning followed by
    FBP. X~2 is the FBP over all 720 views of the restored sinogram; X~1 is the FBP
    of the measured views alone.
  - the learning-rate schedule is not given in the paper: constant 1e-4 here. The best
    validation epoch is kept, and training stops after 15 epochs without a validation
    improvement or 20 h, so most settings will not reach 100 epochs.
  - sinograms enter as s + 1 and images as (label + K) / z_scale, so both are of order 1
* **`common/torch_ct.py`**: the differentiable operators both methods use. `FanFBP`
  reconstructs an ASTRA-projected phantom to within 1e-4 of its true values and its
  backward pass is the exact transpose (checked with gradcheck); on the cluster its
  output differs slightly from ASTRA's FBP_CUDA (other interpolation and filter
  discretization), which DuDoTrans's image network absorbs. torch-radon, which the
  published dual-domain codes depend on, does not build for current GPUs.
* **FBPConvNet**: the original U-Net with a residual connection. Adam 1e-4 with
  cosine decay replaces the paper's SGD.

## Local smoke test

ASTRA has no CPU fan-beam FBP, so without a GPU `common/ct.py` falls back to an
approximate FBP and warns. That's enough to exercise the code, but never report
numbers from it.
