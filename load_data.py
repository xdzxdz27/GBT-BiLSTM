import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from models.GBG import GBDivision

def load_full_df(name: str):
    df = pd.read_csv(f'datasets/{name}.csv')
    df = df[['Open', 'High', 'Low', 'Volume', 'Close']].dropna().reset_index(drop=True)
    return df

def split_train_test(df: pd.DataFrame, test_ratio=0.2):
    total_len = len(df)
    n_train = int(total_len * (1 - test_ratio))
    df_train = df.iloc[:n_train].reset_index(drop=True)
    return df_train, n_train

def build_windows_gb(df_train: pd.DataFrame,
                                       seq_len: int,
                                       gb_cfg: dict,
                                       use_balls_for_global: bool = True): 

    X_raw_full = df_train[['Open', 'High', 'Low', 'Volume']].to_numpy(np.float32)
    y_full     = df_train['Close'].to_numpy(np.float32)
    gb_div = None

    if use_balls_for_global: 
        gb_div = GBDivision(**gb_cfg)
        balls = gb_div.generate(
            df_train,
            feature_cols_idx=slice(0, 4),  # ['Open','High','Low','Volume']
            target_col_idx=4               # 'Close'
        )
        balls_sorted = sorted(balls, key=lambda b: b.start_idx)
        GB_SEQ = np.stack(
            [np.asarray(b.center, dtype=np.float32) for b in balls_sorted],
            axis=0
        )
        GB_RAD = np.asarray(
            [float(b.radius) for b in balls_sorted],
            dtype=np.float32
        )
    else: # A9 
        print("[Ablation A9] Using full raw training sequence as global input.")
        GB_SEQ = X_raw_full # (N_train, 4)
        GB_RAD = None

    X_windows, y_targets = [], []
    N = len(df_train)
    for i in range(N - seq_len):
        Xw = X_raw_full[i:i+seq_len]  # (L,4)
        yt = y_full[i+seq_len]        # predict next Close
        X_windows.append(Xw)
        y_targets.append(yt)

    X_train_raw = np.stack(X_windows, axis=0)
    y_train = np.asarray(y_targets, dtype=np.float32)
    
    return X_train_raw, y_train, GB_SEQ, GB_RAD, gb_div


def build_test_windows(df_full: pd.DataFrame, seq_len: int, train_size: int):
    start = max(0, train_size - seq_len)
    df_test = df_full.iloc[start:].reset_index(drop=True)
    data = df_test[['Open', 'High', 'Low', 'Volume', 'Close']].to_numpy(np.float32)

    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :4])
        y.append(data[i+seq_len, 4])
    return np.stack(X, axis=0), np.asarray(y, dtype=np.float32)

class SeqDatasetSingle(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
