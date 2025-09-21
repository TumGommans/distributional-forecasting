# src/data/dataset.py (Updated)

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


class TabularDataset(Dataset):
    """
    Custom PyTorch Dataset for tabular data.

    It handles one-hot encoding for categorical features and standardizes
    continuous features before combining them into a single input tensor.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        continuous_cols: List[str],
        target_col: str,
        cont_mean: Optional[pd.Series] = None,
        cont_std: Optional[pd.Series] = None
    ):
        """
        Initializes the TabularDataset.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            categorical_cols (List[str]): Column names that are categorical.
            continuous_cols (List[str]): Column names that are continuous.
            target_col (str): The name of the target/outcome column.
            cont_mean (Optional[pd.Series]): Pre-computed means for continuous features.
                                              If None, they will be calculated from df.
            cont_std (Optional[pd.Series]): Pre-computed stds for continuous features.
                                             If None, they will be calculated from df.
        """
        df = df.copy()

        # Target variable

        self.y = torch.tensor(df[target_col].values, dtype=torch.float32).reshape(-1, 1)

        # === Feature Engineering ===
        
        # 1. Standardize continuous features
        if continuous_cols:

            # # Define which of your continuous columns need a log transform
            # # cols_to_log = ["age", "prestg10", "childs"]
            # # Safely find the columns that exist in both your dataframe and the list
            # # This prevents errors if a column name is not found.
            # cols_that_exist = [col for col in cols_to_log if col in df.columns]
            # # Apply log(1+x) transform ONLY to the specified columns
            # if cols_that_exist:
            #     df[cols_that_exist] = np.log1p(df[cols_that_exist])
            if cont_mean is None or cont_std is None:
                # Calculate and store mean/std (for training data)
                self.cont_mean = df[continuous_cols].mean()
                self.cont_std = df[continuous_cols].std()
                # Handle cases where std is zero
                self.cont_std[self.cont_std == 0] = 1.0
            else:
                # Use pre-computed mean/std (for validation/test data)
                self.cont_mean = cont_mean
                self.cont_std = cont_std
            
            # Apply standardization
            df[continuous_cols] = (df[continuous_cols] - self.cont_mean) / self.cont_std
            
        # 2. Process and combine all features
        feature_tensors = []
        if continuous_cols:
            cont_df = df[continuous_cols]
            feature_tensors.append(torch.tensor(cont_df.values, dtype=torch.float32))
        
        if categorical_cols:
            cat_onehot_df = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)
            feature_tensors.append(torch.tensor(cat_onehot_df.values, dtype=torch.float32))

        self.features = torch.cat(feature_tensors, dim=1)
        self.n_records = self.features.shape[0]
        self.input_dim = self.features.shape[1]

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.n_records

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a sample from the dataset at the given index."""
        return self.features[index], self.y[index]