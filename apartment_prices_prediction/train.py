import hydra
import pandas as pd
from catboost import CatBoostRegressor
from dvc.api import DVCFileSystem
from omegaconf import DictConfig


def basic_filter(data):
    return data.drop(["id", "date"], axis=1)


def get_data(data_path):
    fs = DVCFileSystem("./")
    fs.get(data_path, data_path)
    fs.get(data_path, data_path)
    data = pd.read_csv(data_path)
    return basic_filter(data)


@hydra.main(version_base=None, config_path="../configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    train_data = get_data(cfg["train"]["train_data_path"])
    X_train = train_data[train_data.columns[train_data.columns != "price"]]
    y_train = train_data[["price"]]

    cat_features = ["waterfront", "view", "condition", "zipcode"]
    model = CatBoostRegressor(cat_features=cat_features, verbose=False, **cfg["catboost"])
    model.fit(X_train, y_train)

    print("training...")
    model.save_model(cfg["train"]["path_to_save"])
    print(f"model saved to {cfg['train']['path_to_save']}")


if __name__ == "__main__":
    main()
