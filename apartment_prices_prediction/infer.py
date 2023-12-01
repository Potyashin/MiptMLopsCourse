import git
import hydra
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from dvc.api import DVCFileSystem
from mlflow.models import infer_signature
from omegaconf import DictConfig


def basic_filter(data):
    return data.drop(["id", "date"], axis=1)


def get_data(data_path):
    fs = DVCFileSystem("./")
    fs.get(data_path, data_path)
    fs.get(data_path, data_path)
    data = pd.read_csv(data_path)
    return basic_filter(data)


def calculate_metrics(y_true, y_pred):
    """
    Calculates metrics
    """
    metrics = {}
    rmse = np.mean((y_true - y_pred) ** 2) ** 0.5
    mape = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-6))
    max_error = np.max(np.abs(y_true - y_pred))

    metrics = {"RMSE": rmse, "MAPE": mape, "max error": max_error}
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(uri=cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["exp_name"])

    val_data = get_data(cfg["infer"]["val_data_path"])
    X_val = val_data[val_data.columns[val_data.columns != "price"]]
    y_val = val_data[["price"]]

    model = CatBoostRegressor(**cfg["catboost"])
    model.load_model(cfg["infer"]["model_path"])

    y_pred = model.predict(X_val)
    metrics = calculate_metrics(y_val["price"].values, y_pred)

    with mlflow.start_run():
        mlflow.log_params(cfg["catboost"])
        mlflow.log_metrics(metrics)

        repo = git.Repo()  # search_parent_directories=True
        sha = repo.head.object.hexsha
        mlflow.set_tag("commit_id", sha)

        signature = infer_signature(X_val, y_val)
        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path=cfg["infer"]["model_path"],
            signature=signature,
            registered_model_name="catboost_model_1",
        )

    print(metrics)
    pd.DataFrame(y_pred, columns=["prices"]).to_csv(cfg["infer"]["path_to_save_pred"])
    print(f"predictions saved to {cfg['infer']['path_to_save_pred']}")


if __name__ == "__main__":
    main()
