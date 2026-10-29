# OnlineSort

OnlineSort learns a neuronal identity representation from simulated extracellular recordings, initializes identities from an unlabeled prefix, and assigns later events with fixed networks and confidence-gated reference updates. This repository contains the core training, inference and evaluation used by the paper. Data and checkpoints must be obtained separately; they are not downloaded or bundled here.

## Install

Use Python 3.12. From the repository root:

```bash
python -m pip install -r requirements-dev.txt -c constraints-paper.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pytest -q
```

Alternatively, use `conda env create -f environment.gpu.yml`. The paper environment used PyTorch 2.10.0 with CUDA 12.8. Install a matching CUDA-enabled PyTorch build and a compatible NVIDIA driver for training and full inference. CPU inference is supported for small checks; training requires CUDA. `constraints-paper.txt` records the observed training versions. A Conda environment also supplies the C++ runtime required by SciPy wheels; activate the environment before running.

## Data

Training and inference consume MEArec H5 recordings with `recordings` (time × channels), `channel_positions`, sampling frequency `info/recordings/fs`, and MEArec spike/template metadata. GT is used for supervised simulation training and **separate evaluation**, never for target initialization or identity assignment. Multi-channel recordings require a matching electrode geometry; the supplied geometry is Neuronexus-32.

The ordered manifests in `configs/manifests` identify the 36 training recordings and 384 main-benchmark recordings. Training retains a fixed 29/7 recording split. The benchmark includes 192 static and 192 drift recordings. Supply a recordings root with this layout:

```text
recordings/
  train7/hd32_static/*.h5
  train7/hd32_drift/*.h5
  ood_test3/hd32_static/*.h5
  ood_test3/hd32_drift/*.h5
```

The manifest also includes the benchmark segment filenames. Use `--data-root /path/to/recordings`, or place recordings under `data/recordings`. Arbitrary new MEArec recordings can be supplied through the single-recording commands. Matching the paper numbers requires the original cohort and checkpoints; generating different simulations is a separate generalization experiment.

## Import paper checkpoints

Obtain the checkpoints separately, then import them by role:

```bash
python scripts/import_weights.py --role encoder --source /path/to/GlobalIdentity_validation_best.ckpt
python scripts/import_weights.py --role stage1 --source /path/to/stage1_gate_best.ckpt
```

Import verifies SHA256 and architecture before copying to `weights/paper_encoder.ckpt` or `weights/stage1_gate.ckpt`. The expected hashes are in `configs/paper_reference.json`. Different files are rejected. No weights are committed to this repository. A locally trained encoder can instead be passed directly with `--weights`.

## Train

From raw training recordings, train the gate and encoder:

```bash
python scripts/pretrain.py --config configs/train.yaml --data-root /path/to/recordings --dry-run
python scripts/pretrain.py --config configs/train.yaml --data-root /path/to/recordings --device cuda
```

Or start from the imported stage-one checkpoint:

```bash
python scripts/pretrain.py --config configs/train.yaml --data-root /path/to/recordings --stage2-init-from weights/stage1_gate.ckpt --device cuda
```

The gate stage has 20 configured epochs. Encoder training executes 30 epochs with the original 60-epoch warmup–cosine schedule. The objective is support–query NCE plus 0.2 times Identity Discrimination Loss: same-identity compactness and a relative margin against up to 16 nearby different-identity events. Losses are recording-local and computed separately per augmented view. Validation selects by LSM10, then 10-neighbor purity, then lower boundary error. The best encoder checkpoint is `data/checkpoints/onlinesort_train/pretrain_best.ckpt`.

Custom training data can be supplied with `--mearec-h5 file_a.h5 file_b.h5`, or `--patches` and `--labels`. These overrides bypass the paper manifest. Cache files are generated locally. Resume uses `--resume-from` and cannot be combined with `--stage2-init-from`. New training may differ numerically from the released checkpoint; exact paper-result checks use the paper encoder.

## Infer without GT

Single recording:

```bash
python scripts/run_online.py --config configs/infer.yaml --input-h5 /path/to/recording.h5 --weights weights/paper_encoder.ckpt --device cuda --out-npz data/cache/example.npz
```

The prediction file is `data/cache/example_online_init5s.npz`, with a `.run.json` sidecar containing the effective configuration, weight/prediction hashes, environment and diagnostics. Use `--device cpu` for a small CPU check; use `--dry-run` to validate inputs and load the model without processing signal samples. The buffered implementation uses signal context, so computation per candidate is not event-to-label delay.

Batch paper cohort:

```bash
python scripts/run_online.py --config configs/infer.yaml --data-root /path/to/recordings --jobs all --device cuda
```

Pass comma-separated manifest recording names with `--jobs` for a subset. Batch predictions default to `data/cache/onlinesort_infer/<name>_online_init5s.npz`. Existing pure-inference outputs require a new output path or `--force` to overwrite.

Defaults: 5 s unlabeled initialization, gate 0.50, at most five references per identity, distance limit 0.45, gap 0.01, two-hop candidates, update confidence 0.50, update rate 0.05. HDBSCAN candidates and merging use the frozen no-GT selection rules. The initialized identity set remains fixed; rejected events are `-1` and never create identities. Networks remain fixed. The default path does not run full-recording clustering or automatic GT scoring.

## Evaluate saved predictions

Single recording:

```bash
python scripts/evaluate.py --prediction data/cache/example_online_init5s.npz --gt /path/to/recording.h5 --condition drift --output-dir data/reports/example
```

GT may also be a separately prepared **peak-aligned** NPZ containing `times` in samples, `labels`, and optional `channels`. H5 GT is peak-aligned by the existing preprocessing/alignment code. Predictions are loaded and hashed before GT access; evaluation never changes them.

Full paper cohort:

```bash
python scripts/evaluate.py --data-root /path/to/recordings --predictions-dir data/cache/onlinesort_infer --check-reference
```

Both predictions and GT are clipped to `[5,100)` s, with 0.5 ms tolerance and one-to-one identity assignment. All GT spikes remain in the recall denominator, including those with no output. Zero-match recordings remain zero. Scores are averaged equally across recordings; mean F1 is not reconstructed from mean precision and recall. Use `--start-s 20 --stop-s 100` for the common comparison window.

Outputs are `per_record.csv`, `summary.csv`, `comparisons.csv` and `run.json`. Conditional NMI/ARI use only time/channel-matched assigned events; they do not measure missed-spike recovery. Reference means for the original 384-record cohort are overall precision 0.8451, recall 0.6580 and F1 0.7321; static/drift F1 is 0.7623/0.7019. `--check-reference` verifies the complete cohort and compares computed means with these references to tolerance 0.0001. Reference values are never substituted for measured results.

### External predictions

External baseline implementations are not included. Supply a JSON prediction manifest using `--prediction-manifest`; paths are relative to that manifest:

```json
{"records": [
  {"name": "recording_a", "method": "OnlineSort", "condition": "drift", "prediction": "predictions/online_a.npz", "gt": "recordings/a.h5"},
  {"name": "recording_a", "method": "MountainSort4", "condition": "drift", "prediction": "predictions/ms4_a.npz", "gt": "recordings/a.h5"}
]}
```

Prediction NPZ files use `event_time_sample`, `unit_id` and `center_channel`; adapt other formats before scoring. Entries can specify `start_s` and `stop_s` (for example, 20 and 100 for retained E-Sort predictions). Comparisons require identical recording sets, conditions and scoring intervals. Without external predictions, only OnlineSort results are reported, and the reported 24.4% precision improvement is not presented as reproduced.

## License

MIT; see `LICENSE`.
