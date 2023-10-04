import pandas as pd
from catboost import CatBoostRegressor
import numpy as np
import argparse


def basic_filter(data):
    return data.drop(['id', 'date'], axis=1)


def get_data(data_path):
    data = pd.read_csv(data_path)
    return basic_filter(data) 


def calculate_metrics(y_true, y_pred):
    """
    Calculates metrics
    """
    metrics = {}
    rmse = np.mean((y_true - y_pred)**2)**0.5
    mape = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-6))

    metrics['RMSE'] = rmse
    metrics['MAPE'] = mape
    return metrics
    

if __name__ == '__main__':
    
    parser.add_argument('--val_data_path',
                    type=str,
                    default='./data/houses_val.csv')
    
    parser.add_argument('--model_path',
                    type=str,
                    default='./models/model.cbm')
    
    parser.add_argument('--path_to_save_pred',
                    type=str,
                    default='./data/val_pred.csv')
    
    args = parser.parse_args()
    
    val_data = get_data(args.val_data_path)
    X_val = val_data[val_data.columns[val_data.columns != "price"]]
    y_val = val_data[["price"]]
    
    model = CatBoostRegressor()
    model.load_model(args.model_path)
    
    y_pred = model.predict(X_val)
    metrics = calculate_metrics(y_val['price'].values, y_pred)
    print(metrics)
    pd.DataFrame(y_pred, columns=['prices']).to_csv(args.path_to_save_pred)
    print(f'predictions saved to {args.path_to_save_pred}')
    