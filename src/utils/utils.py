"""Script for utility functions."""

import yaml
import json
import numpy as np
import pandas as pd
import hyperopt as hp

from typing import Dict, Any

def load_json(path: str) -> Dict[str, Any]:
    """Load and parse a JSON file.

    Args:
        path: the path to the JSON file

    Returns:
        Dict[str, Any]: the output dictionary
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def load_config(path: str) -> Dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        path: the path to the YAML configuration file

    Returns:
        Dict[str, Any]: the config dictionary
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def fetch_space_from_config(path: str) -> Dict[str, Any]:
    """Read config file and build a hyperopt search space.

    Args:
        config_path: the path to the YAML configuration file

    Returns:
        Dict[str, Any]: the hyperopt search space
    """
    config_dict = load_config(path=path)
    config = config_dict['hyperparam_space']

    space = {}
    for param, settings in config.items():
        dist_type = settings['type']
        args = settings['args']

        if dist_type == 'loguniform':
            args['low'] = np.log(args['low'])
            args['high'] = np.log(args['high'])

        hyperopt_func = getattr(hp.hp, dist_type)
        space[param] = hyperopt_func(param, **args)

    return space

def encode_features(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    categorical_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical features on the input data.

    Args:
        X_train: the train data
        X_val: the validation data
        categorical_cols: list of categorical column names
    
    Returns:
        pd.DataFrame: the encoded train data
        pd.DataFrame: the encoded validation data
    """
    X_train_eng = X_train.copy()
    X_val_eng = X_val.copy()

    train_dummies = pd.get_dummies(X_train_eng[categorical_cols], drop_first=True, dtype=int)
    val_dummies = pd.get_dummies(X_val_eng[categorical_cols], drop_first=True, dtype=int)
    val_dummies = val_dummies.reindex(columns=train_dummies.columns, fill_value=0)

    X_train_eng = X_train_eng.join(train_dummies)
    X_val_eng = X_val_eng.join(val_dummies)

    X_train_eng = X_train_eng.drop(columns=categorical_cols)
    X_val_eng = X_val_eng.drop(columns=categorical_cols)

    return X_train_eng, X_val_eng
