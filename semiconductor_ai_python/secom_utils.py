"""SECOM 真實半導體資料集的共用工具。

教材用途：
Word 文件後半段提到 SECOM dataset，這是一份真實半導體製程資料，特色是：

- 特徵很多，約 590 個製程參數
- 有大量缺失值，符合真實工廠資料常見情況
- label = 1 代表良品，label = -1 代表不良品

這個檔案集中處理「讀檔、缺失值處理、切分資料、標準化」，讓其他 SECOM
範例程式不用重複寫相同流程。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


LABEL_COLUMN = "label"
RANDOM_STATE = 42


def _resolve_input(path_value: Union[str, Path]) -> Path:
    """尋找資料檔案位置。

    程式會依序檢查：
    1. 使用者傳入的路徑
    2. 目前執行指令所在資料夾
    3. 這份程式所在資料夾
    """
    path = Path(path_value)
    candidates = [
        path,
        Path.cwd() / path,
        Path(__file__).resolve().parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {path_value}. Place it next to these scripts or pass an explicit path."
    )


def load_secom_raw(
    data_path: Union[str, Path] = "secom.data",
    labels_path: Union[str, Path] = "secom_labels.data",
) -> pd.DataFrame:
    """讀取原始 SECOM 特徵資料與標籤資料。

    secom.data:
        製程特徵資料，每一欄是一個製程參數。
    secom_labels.data:
        良率標籤，1 代表良品，-1 代表不良品。
    """
    data_file = _resolve_input(data_path)
    labels_file = _resolve_input(labels_path)

    # Step 1：讀取特徵資料。
    # SECOM 使用空白分隔資料；NaN 表示缺失值。
    features = pd.read_csv(
        data_file,
        sep=r"\s+",
        header=None,
        na_values=["NaN"],
        engine="python",
    )

    # Step 2：讀取標籤資料。
    labels = pd.read_csv(labels_file, sep=r"\s+", header=None, engine="python")

    # Step 3：替沒有名稱的特徵欄位補上 feature_0, feature_1, ...
    # 這樣後續畫圖或輸出時會比較容易閱讀。
    features.columns = [f"feature_{i}" for i in range(features.shape[1])]

    # 為了避免 PerformanceWarning (DataFrame is highly fragmented)
    # 在插入新欄位前先建立一個連續的副本。
    features = features.copy()
    features[LABEL_COLUMN] = labels.iloc[:, 0].astype(int).to_numpy()
    return features


def preprocess_secom(
    data_path: Union[str, Path] = "secom.data",
    labels_path: Union[str, Path] = "secom_labels.data",
    max_missing_ratio: float = 0.50,
) -> Tuple[pd.DataFrame, pd.Series]:
    """清理 SECOM 資料。

    處理流程：
    1. 移除缺失值比例過高的欄位
    2. 剩下的缺失值用平均數填補
    3. 回傳 X 特徵與 y 標籤
    """
    raw = load_secom_raw(data_path=data_path, labels_path=labels_path)

    # Step 1：把答案 label 與特徵欄位分開。
    y = raw[LABEL_COLUMN].astype(int)
    X = raw.drop(columns=[LABEL_COLUMN])

    # Step 2：計算每個欄位的缺失值比例。
    # 缺失太多的欄位代表資訊不足，所以先移除。
    missing_ratio = X.isna().mean()
    X = X.loc[:, missing_ratio < max_missing_ratio]

    # Step 3：剩下欄位的缺失值用平均數填補。
    # 這是基礎版本，適合教材示範；實務上也可以改用更進階的補值方法。
    X = X.fillna(X.mean(numeric_only=True))

    return X, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """切分資料並進行標準化。

    回傳：
    - X_train_scaled：標準化後的訓練特徵
    - X_test_scaled：標準化後的測試特徵
    - y_train：訓練標籤
    - y_test：測試標籤
    - scaler：已訓練好的 StandardScaler
    """

    # Step 1：如果每個類別資料量足夠，就使用 stratify 維持類別比例。
    class_counts = y.value_counts()
    stratify = y if len(class_counts) > 1 and class_counts.min() >= 2 else None

    # Step 2：切成訓練集與測試集。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Step 3：標準化。
    # fit 只能用訓練集，測試集只能 transform，避免資料洩漏。
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.to_numpy(),
        y_test.to_numpy(),
        scaler,
    )
