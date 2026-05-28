from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def balls_to_dataframe(balls) -> pd.DataFrame:
    rows = []
    for i, b in enumerate(balls):
        d = b.to_dict()
        d["GB_id"] = i
        rows.append(d)
    cols = ["GB_id","center", "target_mean","radius","members","start_idx","end_idx","slope","trend_purity","reg_purity","size"]
    return pd.DataFrame(rows)[cols]


def _ols_slope(y: np.ndarray, t: Optional[np.ndarray]=None) -> float:
    n = len(y)
    if n <= 1: return 0.0
    if t is None: 
        t = np.arange(n, dtype=np.float64)
    tc = t - t.mean()
    yc = y - y.mean()
    denom = (tc ** 2).sum()
    slope = float((tc * yc).sum() / denom)
    return 0.0 if denom <= 1e-12 else slope

def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    result = 1.4826 * float(np.median(np.abs(x - med)) + 1e-12)
    return result

def _trend_labels(close: np.ndarray, window, zeta) -> np.ndarray:
    n = len(close)
    if n == 0: 
        return np.array([], dtype=np.int8)
    half = window // 2
    slopes = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + (window - half))
        slopes[i] = _ols_slope(close[a:b])
    scale = max(_mad(slopes), 1e-12)
    th = zeta * scale
    d = np.zeros(n, dtype=np.int8)
    d[slopes >=  th] =  1
    d[slopes <= -th] = -1
    return d

def _trend_purity(labels: np.ndarray, idxs: np.ndarray) -> float:
    if len(idxs) == 0: 
        return 0.0
    sub = labels[idxs]
    p_plus  = (sub ==  1).sum() / len(sub)
    p_minus = (sub == -1).sum() / len(sub)
    t_p = float(max(p_plus, p_minus))
    return t_p

def _reg_purity(close: np.ndarray, idxs: np.ndarray, eps: float=1.0) -> float:
    if len(idxs) == 0: 
        return 0.0
    y = close[idxs]
    t = idxs.astype(np.float64)

    k = _ols_slope(y, t=t)
    if len(y) > 0:
        b = y.mean() - k * t.mean()
    else:
        b = 0 
    y_hat = k * t + b 
    delta_y = y - y_hat
    if len(delta_y) > 0:
         s_prime = _mad(delta_y)
    else:
         s_prime = 0

    tube_prime = eps * s_prime
    if tube_prime <= 0: 
        return 1.0
    purity = float((np.abs(delta_y) <= tube_prime).mean())

    return purity


@dataclass
class GranularBall:
    center: np.ndarray
    target_mean: float
    members: List[int]
    radius: float
    slope: float
    trend_purity: float
    reg_purity: float
    size: int
    start_idx: int
    end_idx: int

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["center"] = self.center.tolist()
        return d
    
class GBDivision:
    def __init__(self,
                min_samples: int,
                max_samples: int,
                purity_type: str,
                p_threshold: float,
                reg_eps: float,
                alpha: float,
                trend_window: int,
                trend_zeta: float,
                random_state: 42):
        assert min_samples >= 2
        if max_samples is not None:
            assert max_samples >= min_samples, "max_samples must be >= min_samples"
        assert purity_type in ("trend", "trend_reg")

        self.min_samples = min_samples
        self.max_samples = max_samples
        self.purity_type = purity_type
        self.p_threshold = float(p_threshold)
        self.reg_eps = float(reg_eps)
        self.alpha = alpha
        self.trend_window = trend_window
        self.trend_zeta = trend_zeta
        self.random_state = random_state
        self._df, self._X_raw, self._X_std, self._y, self._dates, self._labels, self._scaler, self._feat_cols = (None,) * 8
        self._vis_interval: Optional[int] = None
        self._vis_dir: Optional[str] = None

    def _create_ball(self, idxs: np.ndarray) -> GranularBall: 
        idxs = np.asarray(sorted(list(idxs)), dtype=np.int32)
        Xr = self._X_raw[idxs]
        Xs = self._X_std[idxs]
        yr = self._y[idxs]
        center_raw = Xr.mean(axis=0) 
        center_std = Xs.mean(axis=0, keepdims=True)
        dis_s = np.linalg.norm(Xs - center_std, axis=1)

        r = float(dis_s.mean())
        t = idxs.astype(np.float64)
        k = _ols_slope(yr, t=t)
        tp = _trend_purity(self._labels, idxs)
        rp = _reg_purity(self._y, idxs, eps=self.reg_eps)

        return GranularBall(
            members=idxs.tolist(), center=center_raw, radius=r, slope=k,
            trend_purity=tp, reg_purity=rp, size=len(idxs),
            target_mean=float(yr.mean()), start_idx=int(idxs.min()), end_idx=int(idxs.max()),
        )
    
    def generate(self, df: pd.DataFrame,
                 feature_cols_idx: slice = slice(1, -1),
                 target_col_idx: int = -1) -> List[GranularBall]:
        df = df.copy()

        feature_cols = df.columns[feature_cols_idx].tolist()
        target_col = df.columns[target_col_idx]

        X_raw = df[feature_cols].astype(float).to_numpy()
        y = df[target_col].astype(float).to_numpy()

        dates = np.arange(len(df))
        scaler = StandardScaler().fit(X_raw)
        X_std = scaler.transform(X_raw)

        self._df, self._feat_cols = df, feature_cols 
        self._X_raw, self._X_std, self._y, self._dates, self._scaler = X_raw, X_std, y, dates, scaler
        self._labels = _trend_labels(y, window=self.trend_window, zeta=self.trend_zeta)

        return self._generate_GBs(df)
    
    def _generate_GBs(self, df: pd.DataFrame) -> List[GranularBall]:
        balls_idxs = []
        queue = [np.arange(len(df), dtype=np.int32)] 
        while queue:
            i = queue.pop(0)
            if self._is_stop_split(i):
                balls_idxs.append(i)
                continue
            b_point = self._find_best_time_split(i)
            if b_point is not None:
                queue.append(i[:b_point])
                queue.append(i[b_point:])
            else:
                balls_idxs.append(i)
        balls = [self._create_ball(idxs) for idxs in balls_idxs]
        balls.sort(key=lambda b: b.start_idx)
        return balls
    
    def _is_stop_split(self, idxs: np.ndarray) -> bool:
        if len(idxs) <= self.min_samples:
            return True
        if len(idxs) >= self.max_samples:
            return False
        if self._ball_purity(idxs) < self.p_threshold:
            return False
        return True
    
    def _find_best_time_split(self, idxs: np.ndarray) -> Optional[int]:
        n = len(idxs)
        b_point = None
        max_purity = self._ball_purity(idxs)

        for s in range(self.min_samples, n - self.min_samples + 1):
            left_idxs, right_idxs = idxs[:s], idxs[s:]
            p_left, p_right = self._ball_purity(left_idxs), self._ball_purity(right_idxs)

            current_p = (len(left_idxs) * p_left + len(right_idxs) * p_right) / n
            if current_p > max_purity:
                max_purity = current_p
                b_point = s

        is_large = (self.max_samples is not None and n > self.max_samples)
        if b_point is None and is_large:
            middle_point = n // 2
            if middle_point >= self.min_samples and (n - middle_point) >= self.min_samples:
                return middle_point
            else:
                return None
    
        return b_point
        
    def _ball_purity(self, idxs: np.ndarray) -> float:
        tp = _trend_purity(self._labels, idxs)
        if self.purity_type == "trend": 
            return tp
        rp = _reg_purity(self._y, idxs, eps=self.reg_eps)
        purity = self.alpha * tp + (1.0 - self.alpha) * rp
        return purity
