import hydra
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from omegaconf import DictConfig


def basic_filter(data):
    return data.drop(["id", "date"], axis=1)


def get_data(data_path):
    data = pd.read_csv(data_path)
    return basic_filter(data)


def calculate_metrics(y_true, y_pred):
    """
    Calculates metrics
    """
    metrics = {}
    rmse = np.mean((y_true - y_pred) ** 2) ** 0.5
    mape = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-6))

    metrics["RMSE"] = rmse
    metrics["MAPE"] = mape
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    val_data = get_data(cfg["infer"]["val_data_path"])
    X_val = val_data[val_data.columns[val_data.columns != "price"]]
    y_val = val_data[["price"]]

    model = CatBoostRegressor(**cfg["catboost"])
    model.load_model(cfg["infer"]["model_path"])

    y_pred = model.predict(X_val)
    metrics = calculate_metrics(y_val["price"].values, y_pred)
    print(metrics)
    pd.DataFrame(y_pred, columns=["prices"]).to_csv(cfg["infer"]["path_to_save_pred"])
    print(f"predictions saved to {cfg['infer']['path_to_save_pred']}")


if __name__ == "__main__":
    main()
