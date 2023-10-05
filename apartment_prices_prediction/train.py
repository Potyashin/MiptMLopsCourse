import pandas as pd
from catboost import CatBoostRegressor
import numpy as np
import argparse


def basic_filter(data):
    return data.drop(['id', 'date'], axis=1)


def get_data(data_path):
    data = pd.read_csv(data_path)
    return basic_filter(data) 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Videos to images')
    
    parser.add_argument('--train_data_path',
                    type=str,
                    default='./data/houses_train.csv')
    
    parser.add_argument('--path_to_save',
                    type=str,
                    default='./models/model.cbm')
    
    args = parser.parse_args()
    
    train_data = get_data(args.train_data_path)
    X_train = train_data[train_data.columns[train_data.columns != "price"]]
    y_train = train_data[["price"]]
    
    cat_features = ['waterfront', 'view', 'condition', 'zipcode']
    model = CatBoostRegressor(cat_features=cat_features, verbose=False)
    model.fit(X_train, y_train)
    
    print('training...')
    model.save_model(args.path_to_save)
    print(f'model saved to {args.path_to_save}')