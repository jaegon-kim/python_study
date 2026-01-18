import os

import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils
from sktime.forecasting.ttm import TinyTimeMixerForecaster


class TritonPythonModel:
    def initialize(self, args):
        self._model_path = os.environ.get("TTM_MODEL_PATH") or None

    def execute(self, requests):
        responses = []
        for request in requests:
            y_tensor = pb_utils.get_input_tensor_by_name(request, "y")
            fh_tensor = pb_utils.get_input_tensor_by_name(request, "fh")

            y = y_tensor.as_numpy().astype(np.float32).ravel()
            fh = fh_tensor.as_numpy().astype(np.int64).ravel().tolist()

            y_series = pd.Series(y, index=pd.RangeIndex(len(y)))

            forecaster = TinyTimeMixerForecaster(model_path=self._model_path)
            forecaster.fit(y_series, fh=fh)
            y_pred = forecaster.predict()

            output = y_pred.to_numpy().astype(np.float32)
            out_tensor = pb_utils.Tensor("y_pred", output)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
