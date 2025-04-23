import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class Bot_Dataset(Dataset):
    # Takes full path to data directory
    def __init__(self, path):
        bot_df = pd.read_parquet(os.path.join(path, 'revised_bot_data.parquet'))
        human_df = pd.read_parquet(os.path.join(path, 'revised_user_data.parquet'))

        df = pd.concat([bot_df, human_df], ignore_index=True)

        # Collapse the columns of DataFrame for better passing into model
        desc_cols = [f"description{i}" for i in range(1, 769)]
        twt_cols = [f"twt_{i}" for i in range(1, 769)]
        name_cols = [f"name{i}" for i in range(1, 769)]

        desc_np = df[desc_cols].values.astype(np.float32) 
        twt_np  = df[twt_cols].values.astype(np.float32)  
        name_np = df[name_cols].values.astype(np.float32) 
        tokens = np.stack([desc_np, twt_np, name_np], axis=-1) # new shape: [n, 768, 3]
        tokens = np.transpose(tokens, (0, 2, 1)) # reshape for ResNet

        df = df.drop(columns = desc_cols + twt_cols + name_cols)
        scalar_features = df.drop(columns='label').values.astype(np.float32)

        # Convert X and y to torch tensors
        self.X = torch.from_numpy(tokens)
        self.y = torch.from_numpy(df['label'].values)
        self.scalar = torch.from_numpy(scalar_features)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.scalar[idx], self.y[idx]]