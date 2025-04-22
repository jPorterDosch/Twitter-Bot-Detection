import pandas as pd
import numpy as np
import matplotlib.pyplot as pltAP
import seaborn as sns
from sklearn import train_test_split
from torch.utils.data import Dataset, DataLoader
import os

class Bot_Dataset(Dataset):
    # Takes full path to data directory
    def __init__(self, path):
        bot_df = pd.read_parquet(os.path.join(path, 'revised_bot_data.parquet'))
        human_df = pd.read_parquet(os.path.join('revised_user_data.parquet'))

        df = pd.concat([bot_df, human_df], ignore_index=True)

        y = df['label']

        X = df.drop(columns='label')

    def __len__(self):
        return len(self.X)


RANDOM_STATE = 42

bot_df = pd.read_parquet('data/revised_bot_data.parquet')
human_df = pd.read_parquet('data/revised_user_data.parquet')

df = pd.concat([bot_df, human_df], ignore_index=True)

y = df['label']

X = df.drop(columns='label')

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)