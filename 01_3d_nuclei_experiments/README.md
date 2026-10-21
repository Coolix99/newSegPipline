# 01 — Experimental 3D nuclei segmentation (code-only)

**Everything bulky is outside Git.** This is an isolated research benchmark; it does not modify production segmentation. No command-line arguments are necessary. Edit `config.yaml`.

## Location on your Linux machine

Configured paths (resolved from `data_root: /media/ssd2/vinita_segmentation`):

| Path | Meaning |
|---|---|
| `examples_gt/` | Manually annotated source patches; **read-only** |
| `examples_segmented_results/` | Existing weak/old *segmentation masks*; **read-only** |
| `experiment_data/weak_patches/` | Optional paired weak training patches; see `docs/WEAK_DATA.md` |
| `experiment_cache/features/` | N5/N6/N7 cached pretrained features |
| `experiment_cache/pretrained_weights/` | Downloaded μSAM / Cellpose weights |
| `experiments/01_3d_nuclei_experiments/` | Training checkpoints, predictions, CSV and debugging figures |

These folders are **never created under this source checkout**. The Python config reader rejects paths located within this script folder. No example images are packaged. The scripts read your existing `examples_gt` directly and use the existing intensity normalization.

You may override `data_root` for another computer using `VSEG_DATA_ROOT=/some/other/path`.

## Get started

With your PyTorch environment activated, run from this code folder:

```bash
python 00_run_tests.py
python 00_check_setup.py
python 01_compare_models.py
python 02_sweep_reconstruction.py
python 03_view_napari.py  # optional
```

`01` automatically runs tests before training. `00_check_setup.py` prints the resolved SSD paths and reports missing feature caches. `01` trains N0–N4 normally and **skips** N5–N7 if their real cached features are missing. Main results are written to `experiments/01_3d_nuclei_experiments/comparison.csv` under the SSD root. A full set of debug figures is generated per model.

**Existing results:** In the previous package, checkpoints were written under `01_3d_nuclei_experiments/outputs/` within the repository. Move that folder to the external output location before overwriting old work:

```bash
mkdir -p /media/ssd2/vinita_segmentation/experiments
mv outputs /media/ssd2/vinita_segmentation/experiments/01_3d_nuclei_experiments
```

Only run `mv` from the **old** experiment folder, and only if the target does not already exist; otherwise merge carefully. Git ignores don't remove already tracked files: if any bulky files were committed, untrack them with `git rm --cached` before committing the cleanup (without deleting the SSD originals).

## Enable the skipped pretrained models (N5/N6/N7)

These were skipped because you have not generated the frozen-encoder embeddings; they are **not failed trainings**.

1. Install μSAM in the current environment: `python -m pip install micro_sam` (N5/N6).
2. For N7, upgrade Cellpose and install DINOv3 as documented by its developers: `python -m pip install -U cellpose` and `python -m pip install 'git+https://github.com/facebookresearch/dinov3'`.
3. To experiment with the current 64×64 XY patches, explicitly set `allow_foundation_tiny_patches: true` in `config.yaml`. Default is **false**, because stretching tiny patches to the foundation encoder's expected input size loses the benefits of large-scale context.
4. Begin with `foundation_encoders: [microsam]`, run `python 04_cache_foundation.py`, then run `python 01_compare_models.py`. With XY/XZ/YZ caches, N5 and N6 can train.
5. Set `models: [N5_microsam_xy, N6_microsam_xyz]` when testing only μSAM (no redundant CNN retraining). Existing comparison rows are preserved.
6. Change to `foundation_encoders: [cpdino]`, rerun `python 04_cache_foundation.py`, then select `models: [N7_cpdino_xy]` and rerun `python 01_compare_models.py`.

For meaningful scientific comparisons, build embeddings from **larger source tiles** and spatially align them with the annotated patches. The current adapter extracts features from the provided patch itself; this is a preliminary comparison only. No ImageNet/SAM weights are bundled in Git, and the optional extraction adapters require verification on your installed micro-sam / Cellpose versions. Cache generation can be slow because each 64³ patch has 64 slices per orientation.

Links: [μSAM documentation](https://computational-cell-analytics.github.io/micro-sam/), [Cellpose installation](https://cellpose.readthedocs.io/en/latest/installation.html), [Cellpose model types](https://cellpose.readthedocs.io/en/latest/models.html).

## Experiment rules

- Predict **foreground + 3D center-diffusion flow** with interchangeable architectures N0–N7; normalized foreground, flow and total losses are logged separately.
- Fixed default reconstruction: 50 steps at 0.2 with nearest-voxel flow sampling, connected components and DBSCAN (`eps=3`, `min_samples=27`). `02` sweeps parameters using cached *validation* predictions.
- The image was already normalized: **no additional normalization**. Intensity-zero voxels are ignored in training and forced to background.
- Augmentation: XY 90° rotations, X flips and Z flips. **Never exchange Z with X/Y.** Flow channels are transformed along with the images.
- Data split is at source-group level. If you know the original specimens, put a JSON mapping of case IDs to specimen IDs at `/media/ssd2/vinita_segmentation/examples_gt/groups.json`; otherwise, a **Git-commit heuristic** is used and is not biologically reliable. The supplied five patches cannot establish a robust architecture ranking.
- Old segmentation result TIFFs have **no matching raw nucleus TIFFs** in the provided archive. Therefore `pretrain_weak: false` until matching images, flow targets and profiles are prepared as paired patches. See `docs/WEAK_DATA.md`.
- Unit tests are synthetic where possible, run without data, and conditionally check available real samples. Pretrained adapter inference cannot be unit tested offline without installing the optional encoders and downloading their weights.

## Git hygiene

This folder includes `.gitignore` for any remnants of the old package (`data`, `outputs`, model weights, figures, caches). In the new version **all data and results are actually outside Git**, so ignore rules are a second line of defense. The package contains code, small JSON metadata, tests and docs only.

## Pretrained 64³ patch trial / Numba compatibility

On the five provided samples, first run `python 00_run_tests.py`, then `python 04_cache_foundation.py` to download pretrained weights and cache XY and XYZ embeddings (this can take time). Run `python 00_check_setup.py` to verify that N5/N6/N7 report READY, then `python 01_compare_models.py`. The default config selects only the previously skipped models so previous N0–N4 results are preserved.

`numba` is optional; when it cannot import because of mismatched `llvmlite`, reconstruction falls back to the same nearest-voxel integration in vectorized NumPy (slower). For production speed, repair the env using a matching Numba/llvmlite pair. `02_sweep_reconstruction.py` uses sweep_* settings; it does not affect model training or baseline segmentation.

### Comparing all eight models / plotting (updated)

`config.yaml` now selects N0–N7 and has `reuse_completed: true`.
Run `python 01_compare_models.py` to train only missing models and consolidate all
completed metrics into the SSD `comparison.csv`. Previously completed models need
`best.pt`, `summary.json`, and `val_predictions/` in their SSD model folders to be
reused. To avoid retraining N0–N4, move their old output directories into the
new external `output_dir` first. The script prints the exact saved plot paths.

Run `python 02_sweep_reconstruction.py` separately to evaluate different propagation
steps, DBSCAN `eps` and `min_samples`. It scans ALL model folders with saved
validation predictions, not only the selected `models` list, and saves:
`reconstruction_comparison.csv`, `reconstruction_by_steps.png` and
`reconstruction_dbscan_heatmaps.png`. The per-model sweep CSV is also saved.
The parameters used in model training/inference stay unchanged by this sweep.

Diagnostics include per-model `learning_curves.png` and `debug/<case>.png` (XY/XZ/YZ
foreground/instances/errors), plus `debug/<case>_motion.png` (short-flow contraction).
`model_comparison.png` and `model_learning_curves.png` summarize all completed models.
Cached validation predictions are sufficient to regenerate representative debug figures
for reused models; no foundation encoder needs to be rerun.
