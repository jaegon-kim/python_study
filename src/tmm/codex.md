# Codex Notes

## TinyTimeMixerForecaster (Airline, zero-shot) local run

### What was set up
- Example script: `scripts/run_ttm_airline_zeroshot.py`
- Local deps list: `requirements.txt` (CPU-only torch + accelerate)
- Triton model repo: `triton_model_repository/ttm/` (Python backend)

### Local run commands
```bash
cd /root/Workspace/python_study/src/tmm
python3 -m pip install --break-system-packages -r requirements.txt
python scripts/run_ttm_airline_zeroshot.py
```

### Notes
- First run downloads the Hugging Face model `ibm/TTM` (requires network).

## Triton/KServe (venv-based runtime packaging)

### Current state (rebuild after Codex restart)
- Model repo: `triton_model_repository/ttm/`
- Venv-based execution env for Triton Python backend is used.
- `triton_model_repository/ttm/config.pbtxt` includes:
  - `parameters` with `EXECUTION_ENV_PATH = "/mnt/models/ttm/1/venv"`
- Python 3.10 installed via deadsnakes PPA to match Triton 23.10 (Python 3.10).
- venv created and populated at:
  - `triton_model_repository/ttm/1/venv`
- Bundled `python/` dependency folder was removed to avoid conflicting import paths:
  - removed `triton_model_repository/ttm/1/python`
- Updated archive (to be served to KServe):
  - `/root/Workspace/python_study/src/tmm/triton_model_repository.tar.gz`
  - Latest infer works; response example:
    - `/v2/models/ttm/infer` returns `y_pred` with 3 values.
- Make targets:
  - `make triton` builds venv + tar
  - `make clean` removes tar
  - `make clean-all` removes tar + venv + bundled model

### Commands used (for reproducibility)
```bash
# Rebuild venv + tar in one step
sudo bash /root/Workspace/python_study/src/tmm/scripts/build_triton_venv.sh

# Or run commands manually:
# Install Python 3.10 + venv support (Ubuntu 24.04)
apt-get update
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.10-venv

# Create venv and install model deps
rm -rf /root/Workspace/python_study/src/tmm/triton_model_repository/ttm/1/venv
python3.10 -m venv /root/Workspace/python_study/src/tmm/triton_model_repository/ttm/1/venv
/root/Workspace/python_study/src/tmm/triton_model_repository/ttm/1/venv/bin/pip install \
  -r /root/Workspace/python_study/src/tmm/triton_model_repository/ttm/1/requirements.txt

# Remove bundled python deps (if present)
rm -rf /root/Workspace/python_study/src/tmm/triton_model_repository/ttm/1/python

# Rebuild tar to serve to KServe
tar -czf /root/Workspace/python_study/src/tmm/triton_model_repository.tar.gz \
  -C /root/Workspace/python_study/src/tmm/triton_model_repository ttm
```

### Notes
- Triton 23.10 uses Python 3.10; venv must be built with Python 3.10 or pandas/ABI errors occur.
- Serve the tar via HTTP and ensure KServe can fetch it (storage-initializer expects a tar/gzip Content-Type).
- `build_triton_venv.sh` downloads the Hugging Face model snapshot (`ibm/TTM`) into `ttm/1/ttm_model` for offline use.
- Inference call (Triton V2 HTTP):
  - Inputs: `y` (FP32, len 12), `fh` (INT64, len 3)
  - Output: `y_pred` (FP32, len 3)
- Model runtime tweaks in `triton_model_repository/ttm/1/model.py`:
  - Add venv site-packages to `sys.path`
  - Set HF cache env vars to `/tmp/huggingface`
  - Use bundled model path at `ttm/1/ttm_model` when `TTM_MODEL_PATH` is unset
  - Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` when bundled model is used
  - If `TTM_MODEL_PATH` is not set, use `fit_strategy="full"`
  - Pass `training_args={"output_dir": "/tmp/ttm_train"}`
