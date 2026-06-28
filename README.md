# Onlinesort

Onlinesort contains the core training, inference, and evaluation code for the online spike sorting pipeline.

## Setup

Install the Python dependencies listed in `requirements-base.txt`. A CUDA PyTorch build is required for GPU training and inference.

```bash
pip install -r requirements-base.txt
```

## Training

Edit `configs/train.yaml` and fill `data.mearec_h5_paths` with your own MEArec H5 recordings.

```bash
python scripts/pretrain.py --config configs/train.yaml
```

The trained checkpoint is written to the `train.save` path in the config.

## Online Evaluation

For batch online evaluation, edit `batch.jobs` in `configs/infer.yaml`.

```bash
python scripts/run_online.py --config configs/infer.yaml --jobs all
```

Generated data, weights, logs, reports, and figures are intentionally ignored by git.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
