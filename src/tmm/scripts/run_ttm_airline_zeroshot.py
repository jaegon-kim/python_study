from sktime.datasets import load_airline
from sktime.forecasting.ttm import TinyTimeMixerForecaster


def main() -> None:
    y = load_airline()
    fh = [1, 2, 3]

    forecaster = TinyTimeMixerForecaster()
    # Zero-shot forecasting with default configuration.
    forecaster.fit(y, fh=fh)
    y_pred = forecaster.predict()

    print("forecast_horizon:", fh)
    print(y_pred)


if __name__ == "__main__":
    main()
