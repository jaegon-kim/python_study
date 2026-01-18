import os
import sys
from pathlib import Path

import triton_python_backend_utils as pb_utils
import numpy as np

# Delay heavy imports until after sys.path is patched in initialize.


class TritonPythonModel:
    def initialize(self, args):
        # Ensure HF/Transformers caches use writable paths inside the container.
        os.environ.setdefault("HF_HOME", "/tmp/huggingface")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface/transformers")
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/huggingface/hub")
        venv_dir = Path(__file__).resolve().parent / "venv"
        if venv_dir.exists():
            for site_packages in venv_dir.glob("lib/python*/site-packages"):
                sys.path.insert(0, str(site_packages))
                break
        deps_dir = Path(__file__).resolve().parent / "python"
        if deps_dir.exists():
            sys.path.insert(0, str(deps_dir))
        self._model_path = os.environ.get("TTM_MODEL_PATH") or None

    def execute(self, requests):
        responses = []
        for request in requests:
            # Import after sys.path update to use bundled dependencies.
            import pandas as pd
            from sktime.forecasting.ttm import TinyTimeMixerForecaster

            y_tensor = pb_utils.get_input_tensor_by_name(request, "y")
            fh_tensor = pb_utils.get_input_tensor_by_name(request, "fh")

            y = y_tensor.as_numpy().astype(np.float32).ravel()
            fh = fh_tensor.as_numpy().astype(np.int64).ravel().tolist()

            y_series = pd.Series(y, index=pd.RangeIndex(len(y)))

            training_args = {"output_dir": "/tmp/ttm_train"}
            if self._model_path:
                forecaster = TinyTimeMixerForecaster(
                    model_path=self._model_path,
                    training_args=training_args,
                )
            else:
                forecaster = TinyTimeMixerForecaster(
                    fit_strategy="full",
                    training_args=training_args,
                )
            forecaster.fit(y_series, fh=fh)
            y_pred = forecaster.predict()

            output = y_pred.to_numpy().astype(np.float32)
            out_tensor = pb_utils.Tensor("y_pred", output)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
