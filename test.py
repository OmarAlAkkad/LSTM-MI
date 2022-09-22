import pickle
import numpy as np
import tensorflow as tf
import statistics
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import pandas as pd


def load_data(year):
    data = pickle.load(open(f'dataset_year_{year}_scaled.p', 'rb'))

    return data

def to_sequences(data, seq_size):
    x = []
    y = []

    for i in range(len(data) - seq_size):
        x.append(data[i:(i + seq_size), 0])
        y.append(data[i+seq_size, 0])

    return np.array(x), np.array(y).reshape(-1,1)


def rmse(predicted, actual):
    rmse = (np.sqrt(mean_squared_error(actual[:,0], predicted[:,0])))/statistics.mean(actual[:,0])

    return rmse

if __name__ == "__main__":
    percent = 0.5
    epoch = 100

    years = 4
    past_days = 7

    model = keras.models.load_model(f'shadow_0_{percent}_{epoch}')
    scaler = pickle.load(open('scaler.p', 'rb'))
    models = []
    train = []
    test = []
    for year in range(years):
        locals()[f'dataset_year_{year}'] = load_data(year)
        locals()[f'train_year_{year}'], locals()[f'labels_year_{year}'] = to_sequences(locals()[f'dataset_year_{year}'], past_days)
        locals()[f'predictions_year_{year}'] = model.predict(locals()[f'train_year_{year}'])
        locals()[f'predictions_year_{year}'] = scaler.inverse_transform(locals()[f'predictions_year_{year}'])
        locals()[f'labels_year_{year}'] = scaler.inverse_transform(locals()[f'labels_year_{year}'])

        locals()[f'train_score_year_{year}'] = rmse(locals()[f'predictions_year_{year}'], locals()[f'labels_year_{year}'])

        models.append(f'Year_{year}')
        train.append(locals()[f'train_score_year_{year}'] )

    model = keras.models.load_model(f'target_{percent}_{epoch}')
    for year in range(years):
        locals()[f'dataset_year_{year}'] = load_data(year)
        locals()[f'train_year_{year}'], locals()[f'labels_year_{year}'] = to_sequences(locals()[f'dataset_year_{year}'], past_days)
        locals()[f'predictions_year_{year}'] = model.predict(locals()[f'train_year_{year}'])
        locals()[f'predictions_year_{year}'] = scaler.inverse_transform(locals()[f'predictions_year_{year}'])
        locals()[f'labels_year_{year}'] = scaler.inverse_transform(locals()[f'labels_year_{year}'])

        locals()[f'train_score_year_{year}'] = rmse(locals()[f'predictions_year_{year}'], locals()[f'labels_year_{year}'])
        test.append(locals()[f'train_score_year_{year}'] )

    df = pd.DataFrame(
            {'Year': models, 'Shadow model trained on year 3 RMSE': train, 'Target model trained on year 0 RMSE': test}
        )

    df.to_csv('checking models.csv')