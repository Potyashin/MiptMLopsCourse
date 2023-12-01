poetry shell
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080  # or other tracking_uri
poetry run mlflow models serve -m models:/catboost_model_1/Staging --no-conda  -h 127.0.0.1 -p 8001  # or other model/stage
