from __future__ import annotations
import os
import io
import numpy as np
import pandas as pd
import chardet
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def _read_any(path: str, file_type: str | None) -> pd.DataFrame:
    ext = (file_type or os.path.splitext(path)[1][1:]).lower()
    if ext == "xlsx":
        return pd.read_excel(path)
    elif ext in ("csv", "data"):
        with open(path, "rb") as f:
            raw = f.read()
        enc = chardet.detect(raw).get("encoding") or "utf-8"
        return pd.read_csv(io.BytesIO(raw), encoding=enc)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def preprocess_dataframe(
    df: pd.DataFrame,
    target: str,
    drop_cols: List[str],
    binary_map: dict,
    target_map: dict,
    unknown_fill_cols: List[str],
    extreme_clip_cols: List[str],
    clip_p: float,
    month_col: str,
    month_map: dict,
    engineered_flags: dict,
    onehot_like_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # 1) 丢弃无关列
    df = df.drop(columns=drop_cols, errors="ignore")

    # 2) 目标映射为 0/1
    if df[target].dtype == object:
        df[target] = df[target].map(target_map)

    # 3) 处理未知/缺失（job/education）
    for col in unknown_fill_cols:
        if col in df.columns:
            mode_val = df[col].mode(dropna=True)
            if len(mode_val) > 0:
                df[col] = df[col].replace(["unknown", "nonexistent"], mode_val[0])

    # 4) 二元列映射
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # 5) 极值修剪与负值调整
    for col in extreme_clip_cols:
        if col in df.columns:
            df[col] = np.where(df[col] < 0, 0, df[col])
            p = df[col].quantile(clip_p)
            df[col] = np.where(df[col] > p, p, df[col])

    # 6) 月份映射
    if month_col in df.columns:
        df[month_col] = df[month_col].map(month_map)

    # 7) 工程特征
    if engineered_flags.get("debt_ratio", True) and set(["balance", "age"]).issubset(df.columns):
        df["debt_ratio"] = df["balance"] / (df["age"] + 1e-5)

    if engineered_flags.get("credit_usage", True) and "loan" in df.columns and "balance" in df.columns:
        df["credit_usage"] = np.where(df["loan"] == 1, df["balance"] * 0.3, df["balance"] * 0.1)

    if engineered_flags.get("response_ratio", True) and set(["previous", "campaign"]).issubset(df.columns):
        df["response_ratio"] = df["previous"] / (df["campaign"] + 1e-5)

    if engineered_flags.get("high_value_target", True) and "balance" in df.columns and "age" in df.columns:
        df["high_value_target"] = np.where(
            (df["balance"] > df["balance"].quantile(0.75)) & (df["age"] > 40), 1, 0
        )

    if engineered_flags.get("season", True) and month_col in df.columns:
        season_map = {1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:0}
        df["season"] = df[month_col].map(season_map)

    if engineered_flags.get("age_balance_interaction", True) and set(["age", "balance"]).issubset(df.columns):
        df["age_balance_interaction"] = df["age"] * df["balance"]

    if engineered_flags.get("job_edu_encoded", True) and set(["job", "education"]).issubset(df.columns):
        df["job_edu"] = df["job"].astype(str) + "_" + df["education"].astype(str)
        le_tmp = pd.Series(pd.factorize(df["job_edu"])[0], index=df.index)
        df["job_edu_encoded"] = le_tmp.values

    if engineered_flags.get("age_group", True) and "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=5, labels=False)

    if engineered_flags.get("balance_group", True) and "balance" in df.columns:
        df["balance_group"] = pd.cut(df["balance"], bins=5, labels=False)

    if engineered_flags.get("duration_min", True) and "duration" in df.columns:
        df["duration_min"] = df["duration"] / 60.0

    if engineered_flags.get("log_duration", True) and "duration" in df.columns:
        df["log_duration"] = np.log1p(df["duration"])

    # 8) 类别列编码
    feature_cols: List[str] = []
    for col in onehot_like_cols:
        if col in df.columns:
            df[col + "_encoded"] = pd.factorize(df[col])[0]
            feature_cols.append(col + "_encoded")

    # 9) 数值列集合（包含工程特征）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    # 将一并放入特征列表（去重保持顺序）
    seen = set()
    for c in numeric_cols + feature_cols:
        if c not in seen:
            seen.add(c)
    feature_cols = list(seen)

    X = df[feature_cols].copy()
    y = df[target].values

    # 10) 缺失填充
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            if np.isnan(med):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(med)

    return X.values, y, feature_cols


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool,
    feature_range=(-0.5, 0.5),
):
    strat = y if stratify else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    strat2 = y_train_val if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=strat2
    )
    scaler = MinMaxScaler(feature_range=feature_range)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def maybe_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    enabled: bool,
    minority_ratio: float,
    k_neighbors_cap: int,
    random_state: int,
):
    if not enabled:
        return X_train, y_train
    # 仅针对二分类少数类示意
    uniq, cnts = np.unique(y_train, return_counts=True)
    if len(uniq) != 2:
        return X_train, y_train
    n_total = len(y_train)
    target_min = int(n_total * minority_ratio)
    # 自动推断少数类
    minority_class = uniq[np.argmin(cnts)]
    k_neighbors = min(k_neighbors_cap, max(1, int(cnts[uniq == minority_class][0]) - 1))
    sm = SMOTE(
        sampling_strategy={minority_class: target_min},
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res