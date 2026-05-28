import math
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.baseModel import (
    BiLSTM_Baseline, 
)
from utils.seed import  set_seed
set_seed(22)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def build_model(model_type, input_dim, seq_len, hidden_dim):
    model_type = model_type.upper()
    ZOO = {
        'BILSTM':    lambda: BiLSTM_Baseline(input_dim=input_dim, hidden_dim=hidden_dim),
    }
    if model_type not in ZOO:
        raise ValueError(f"Unknown model_type={model_type}. Available: {list(ZOO.keys())}")
    return ZOO[model_type]()
def load_stock_csv(name, seq_len=31):
    df = pd.read_csv(f'datasets/{name}.csv')
    df = df[['Open', 'High', 'Low',  'Volume', 'Close']].dropna().reset_index(drop=True)
    data = df.values.astype(np.float32)
    X, y = [], []
    for i in range(len(data) - seq_len):
        seq = data[i:i + seq_len, : 4] 
        target = data[i + seq_len, 4] 
        X.append(seq)
        y.append(target)
    return np.stack(X, axis=0), np.array(y, dtype=np.float32)


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_and_evaluate_baseline(name='str',
                                seq_len=31,
                                test_ratio=0.2,
                                hidden_dim=64,
                                batch_size=128,
                                lr=1e-3,
                                n_epochs=100,
                                model_type='str'):
    print(f"\n===== Training {model_type} Baseline on {name} =====")
    print("="*50)

    X, y = load_stock_csv(name, seq_len)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    # print(f"Sample X[0]: {X[0]}, y[0]: {y[0]}")
    N = len(X)
    print(f"[{name}] Loaded {N} sequences, feature_dim={X.shape[2]}")


    n_train = int(N * (1 - test_ratio))
    X_train_raw, X_test_raw = X[:n_train], X[n_train:]
    y_train_raw, y_test_raw = y[:n_train], y[n_train:]

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw.reshape(n_train, -1)).reshape(X_train_raw.shape)
    X_test  = scaler_X.transform(X_test_raw.reshape(N - n_train, -1)).reshape(X_test_raw.shape)

    y_mean, y_std = y_train_raw.mean(), y_train_raw.std()
    y_train = (y_train_raw - y_mean) / y_std
    y_test  = (y_test_raw  - y_mean) / y_std

    print("-----------------------------------------------------")
    print(f"Train data: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data: X={X_test.shape}, y={y_test.shape}")


    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True) # False???
    test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=batch_size, shuffle=False)


    model = build_model(model_type, input_dim=X.shape[2], seq_len=seq_len, hidden_dim=hidden_dim).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Trainging loop
    train_losses, val_losses = [], []
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(Xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # evaluation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                y_pred = model(Xb)
                loss = criterion(y_pred, yb)
                val_running += loss.item()
        val_loss = val_running / len(test_loader)
        val_losses.append(val_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{n_epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    # test evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            y_pred = model(Xb)
            preds.extend(y_pred.cpu().numpy().flatten())
            trues.extend(yb.numpy().flatten())

    preds, trues = np.array(preds), np.array(trues)
    preds_real = preds * y_std + y_mean
    trues_real = trues * y_std + y_mean

    mae = np.mean(np.abs(preds_real - trues_real))
    rmse = math.sqrt(np.mean((preds_real - trues_real) ** 2))
    R2 = r2_score(trues_real, preds_real)
    mape = np.mean(np.abs((trues_real - preds_real) / trues_real)) * 100
    print(f"[{model_type}] Final Test MAE={mae:.6f}, RMSE={rmse:.6f}, R2={R2:.6f}, MAPE={mape:.6f}%")

    return {
        'Dataset': name,
        'Model': model_type,
        'MAE': round(mae, 6),
        'RMSE': round(rmse, 6),
        'R2': round(R2, 6),
        'MAPE': round(mape, 6),
    }

def run_baseline(data_name_list, models_to_run=None):
    if models_to_run is None:
        models_to_run = [
            'BILSTM'
            ]

    all_metrics = []
    for name in data_name_list:
        for m in models_to_run:
            metric = train_and_evaluate_baseline(name=name, model_type=m, n_epochs=100)
            if metric:
                all_metrics.append(metric)

    if not all_metrics:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(all_metrics)
    cols = [
        'Dataset', 'Model',
        'MAE', 'RMSE', 'R2', 'MAPE'
    ]
    metrics_df = metrics_df[[c for c in cols if c in metrics_df.columns]]

    print("\n" + "="*50)
    print("Baseline Done!")
    print("="*50)
    print(metrics_df)

    return metrics_df
