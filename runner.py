import os
import math
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.Model import GBFusion
from load_data import SeqDatasetSingle,load_full_df,split_train_test,build_windows_gb,build_test_windows      
from utils.seed import  set_seed 
set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_gate_records(gate_dir, data_name, run_id, gate_df):
    os.makedirs(gate_dir, exist_ok=True)
    out_path = os.path.join(gate_dir, f"{data_name}_{run_id}_gate.csv")
    gate_df.to_csv(out_path, index=False)
    print(f"[Save Gate] Saved {run_id} gate stats to {out_path}")
    return out_path


def save_gate_full_records(gate_dir, data_name, run_id, gate_full_df):
    os.makedirs(gate_dir, exist_ok=True)
    out_path = os.path.join(gate_dir, f"{data_name}_{run_id}_gate_full.csv")
    gate_full_df.to_csv(out_path, index=False)
    print(f"[Save Gate Full] Saved {run_id} full gate matrix to {out_path}")
    return out_path


def _primary_prediction(out_dict):
    return out_dict["y_s"]


def run_gbfusion_experiment(
    run_id: str = 'str',
    data_name: str = 'data_name',
    n_epochs: int = 100,
    global_encoder: str = 'str',
    use_radius_pe: bool = True,
):
    seq_len = 31
    batch_size = 128
    lr = 1e-3
    lstm_hidden = 64
    d_model = 64
    nhead = 4
    gate_tau = 2.0
    gate_lambda = 0.2
    lambda_distill = 0.1
    attn_heads = 4
    attn_temp = 1.0

    print(f"\n========= Training [{run_id}] on {data_name} =========")
    print(f"Params: fusion=gate, pool=attn, encoder={global_encoder}, gate_mono=struct, L_distill={lambda_distill}, A_epochs={A_epochs}, B_epochs={B_epochs}")
    print("="*50)

    try:
        df = load_full_df(data_name)
    except FileNotFoundError:
        print(f"[GBFusion] Dataset not found, skipping: datasets/{data_name}.csv")
        return None
        
    df_train, n_train = split_train_test(df, test_ratio=0.2)
    
    if len(df_train) <= seq_len:
        print(f"[GBFusion] Not enough data for seq_len={seq_len} in {data_name}, skipping.")
        return None

    gb_cfg = dict(
        alpha=0.8, max_samples=5, min_samples=2,
        purity_type="trend_reg", p_threshold=1.0, trend_window=3,
        trend_zeta=1.0, reg_eps=0.8, random_state=42
    )
    
    X_train_raw, y_train, GB_SEQ, GB_RAD, gb_div = build_windows_gb(
        df_train, seq_len=seq_len, gb_cfg=gb_cfg,
        use_balls_for_global=True)

    X_test_raw, y_test = build_test_windows(df, seq_len=seq_len, train_size=n_train)

    scaler_feat = StandardScaler()
    scaler_feat.fit(df_train[['Open', 'High', 'Low', 'Volume']].to_numpy(np.float32))
    X_train_scaled = scaler_feat.transform(X_train_raw.reshape(-1, 4)).reshape(X_train_raw.shape)
    X_test_scaled  = scaler_feat.transform(X_test_raw.reshape(-1, 4)).reshape(X_test_raw.shape)

    GB_SEQ_scaled  = scaler_feat.transform(GB_SEQ) 

    if GB_RAD is not None:
        GB_RAD_arr = np.asarray(GB_RAD, dtype=np.float32)
        GB_RAD_tensor = torch.from_numpy(GB_RAD_arr).float().to(DEVICE)
    else:
        GB_RAD_tensor = None

    GB_SEQ_tensor = torch.from_numpy(GB_SEQ_scaled).float().to(DEVICE)

    y_mean, y_std = y_train.mean(), y_train.std() + 1e-6
    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled  = (y_test  - y_mean) / y_std

    train_loader = DataLoader(SeqDatasetSingle(X_train_scaled, y_train_scaled),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(SeqDatasetSingle(X_test_scaled, y_test_scaled),
                              batch_size=batch_size, shuffle=False)

    model = GBFusion(
    input_dim=4, lstm_hidden=lstm_hidden, d_model=d_model, nhead=nhead,
    num_layers=2,
    attn_heads=attn_heads, attn_temp=attn_temp, gate_tau=gate_tau,
    gate_lambda=gate_lambda,
    use_radius_pe=use_radius_pe, global_encoder=global_encoder).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    max_pe_len = getattr(model.pos, 'pe', torch.zeros(1,8192,1)).size(1)
    if GB_SEQ_scaled.shape[0] > max_pe_len:
        print(f"[Warning A9] Global seq ({GB_SEQ_scaled.shape[0]}) > max_pos_embed ({max_pe_len}). Truncating.")
        GB_SEQ_scaled = GB_SEQ_scaled[:max_pe_len, :]
        if GB_RAD_tensor is not None:
            GB_RAD_tensor = GB_RAD_tensor[:max_pe_len]   

    GB_SEQ_tensor = torch.from_numpy(GB_SEQ_scaled).float().to(DEVICE)

    for epoch in range(1, n_epochs + 1):
        model.train()
        
        if A_epochs > 0:
            lam_d = lambda_distill * min(1.0, epoch / max(1, A_epochs))
        else:
            lam_d = lambda_distill
            
        if B_epochs > 0:
            tau_now = max(1.0, gate_tau * (1.0 - (epoch - 1) / max(1, B_epochs)))
        else:
            tau_now = gate_tau

        running = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            if use_radius_pe and (GB_RAD_tensor is not None):
                z_global_vec = model.encode_global_seq(GB_SEQ_tensor, r_seq=GB_RAD_tensor)
            else:
                z_global_vec = model.encode_global_seq(GB_SEQ_tensor)
            optimizer.zero_grad()
            out = model(Xb, z_global_vec=z_global_vec,
            gate_tau_override=tau_now,
            gate_lambda_override=None,
            return_gate_aux=False)
            y_gl, y_s = out["y_gl"], out["y_s"]
            loss_sup = criterion(y_gl, yb) + criterion(y_s, yb)
            loss_distill = lam_d * criterion(y_s, y_gl.detach())
            loss = loss_sup + loss_distill

            loss.backward()
            optimizer.step()
            running += float(loss.item())

        train_loss = running / max(1, len(train_loader))
        train_losses.append(train_loss)
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            z_global_eval = None

            for Xb, yb in test_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                out = model(Xb, z_global_vec=z_global_eval)
                y_pred = _primary_prediction(out)

                loss = criterion(y_pred, yb)
                val_running += float(loss.item())

        val_loss = val_running / max(1, len(test_loader))
        val_losses.append(val_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{n_epochs} | train={train_loss:.6f} | val={val_loss:.6f} | lam_d={lam_d:.3f} | tau={tau_now:.2f}")

    model.eval()
    preds, trues = [], []
    gate_mean_records, gate_std_records = [], []
    local_weight_records, global_weight_records = [], []
    ang_records = []
    gate_vector_records = []
    capture_gate = True
    with torch.no_grad():
        if use_radius_pe and (GB_RAD_tensor is not None):
            z_global_eval = model.encode_global_seq(GB_SEQ_tensor, r_seq=GB_RAD_tensor)
        else:
            z_global_eval = model.encode_global_seq(GB_SEQ_tensor)

        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            out = model(
                Xb,
                z_global_vec=z_global_eval,
                return_gate_aux=capture_gate,
            )
            y_pred = _primary_prediction(out)

            preds.extend(y_pred.cpu().numpy().flatten())
            trues.extend(yb.cpu().numpy().flatten())

    preds = np.array(preds)
    trues = np.array(trues)
    preds_real = preds * y_std + y_mean
    trues_real = trues * y_std + y_mean

    mae = np.mean(np.abs(preds_real - trues_real))
    rmse = math.sqrt(np.mean((preds_real - trues_real) ** 2))
    R2 = r2_score(trues_real, preds_real)
    mape = np.mean(np.abs((trues_real - preds_real) / trues_real)) * 100
    print(f"[{run_id}] Final Test MAE={mae:.6f}, RMSE={rmse:.6f}, R2={R2:.6f}, MAPE={mape:.6f}%")

    print(f"[{run_id}] Saving predictions...")
    pred_dir = "results/preds"
    os.makedirs(pred_dir, exist_ok=True)
    out_path = os.path.join(pred_dir, f"{data_name}_{run_id}.csv")

    trues_flat = np.array(trues_real).flatten()
    preds_flat = np.array(preds_real).flatten()

    ds_index = np.arange(len(trues_flat))

    pred_df = pd.DataFrame({
        'ds': ds_index, 
        'y_true': trues_flat, 
        'y_pred': preds_flat
    })
    pred_df.to_csv(out_path, index=False)
    print(f"[Save Preds] Saved {run_id} preds to {out_path}")

    gate_len = len(trues_flat)
    gate_available = capture_gate and (len(gate_mean_records) == gate_len)

    gate_df = pd.DataFrame({
        'ds': ds_index,
        'y_true': trues_flat,
        'y_pred': preds_flat,
    })
    gate_dir = os.path.join("Addresults", "GateResults", "raw")
    save_gate_records(gate_dir, data_name, run_id, gate_df)

    if gate_available and gate_vector_records:
        gate_matrix = np.vstack(gate_vector_records).astype(np.float32)
        gate_cols = {
            f"g_{idx:03d}": gate_matrix[:, idx]
            for idx in range(gate_matrix.shape[1])
        }
        gate_full_df = pd.DataFrame({
            'ds': ds_index,
            'y_true': trues_flat,
            'y_pred': preds_flat,
            **gate_cols,
        })
    else:
        gate_full_df = pd.DataFrame({
            'ds': ds_index,
            'y_true': trues_flat,
            'y_pred': preds_flat,
        })
    gate_full_dir = os.path.join("Addresults", "GateResults", "full")
    save_gate_full_records(gate_full_dir, data_name, run_id, gate_full_df)

    return {
        'Dataset': data_name,
        'Model': run_id,
        'MAE': round(mae, 6),
        'RMSE': round(rmse, 6),
        'R2': round(R2, 6),
        'MAPE': round(mape, 6),
    }
