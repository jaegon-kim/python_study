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
