"""Main script for training the final best model and making predictions."""

import pandas as pd
import numpy as np

from ngboost import NGBRegressor
from ngboost.distns import LogNormal
from ngboost.scores import LogScore

from src.utils.utils import (
    load_config,
    load_json, 
    encode_features
)

cfg = load_config(path="/workspace/src/ngb/config/config.yml")
hparams = load_json(path="/workspace/results/hparams/best_hparams_ngb.json")

df_x = pd.read_csv("data/X_trn.csv")
df_y = pd.read_csv("data/y_trn.csv")
df = pd.concat([df_y, df_x], axis=1)

df_x_test = pd.read_csv("data/X_test.csv")

df['educcat'] = df['educcat'].map(cfg['education_mapping'])
df_x_test['educcat'] = df_x_test['educcat'].map(cfg['education_mapping'])

X = df.drop('realrinc', axis=1)
y = df['realrinc']

X_train_eng, X_test_eng = encode_features(X, df_x_test, cfg['all_cat_cols'])

ngb = NGBRegressor(
    Dist=LogNormal,
    Score=LogScore,
    n_estimators=hparams['n_estimators'],
    learning_rate=hparams['learning_rate'],
    col_sample=hparams['col_sample'],
    minibatch_frac=hparams['minibatch_frac'],
    random_state=42,
)

ngb.fit(X_train_eng, y)
y_pred_dist = ngb.pred_dist(X_test_eng)

s_params = y_pred_dist.params['s']
scale_params = y_pred_dist.params['scale']

mu_params = np.log(scale_params)
sigma_params = s_params

forecasts = np.random.lognormal(
    mean=mu_params, 
    sigma=sigma_params, 
    size=(1000, X_test_eng.shape[0])
).T

print(f"\n Forecasts have shape: {forecasts.shape}")

np.save("results/predictions/predictions.npy", forecasts)

print("\n Results successfully saved!")
