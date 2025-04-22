import pandas as pd
import torch
from torch.utils.data import Dataset
import os

class Bot_Dataset(Dataset):
    # Takes full path to data directory
    def __init__(self, path):
        bot_df = pd.read_parquet(os.path.join(path, 'revised_bot_data.parquet'))
        human_df = pd.read_parquet(os.path.join(path, 'revised_user_data.parquet'))

        df = pd.concat([bot_df, human_df], ignore_index=True)

        # Convert X and y to torch tensors
        self.y = torch.from_numpy(df['label'].values)

        self.X = torch.from_numpy((df.drop(columns='label')).values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx].unsqueeze(0), self.y[idx]]