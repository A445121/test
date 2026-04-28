"""模擬半導體製程資料的共用工具。

教材用途：
這個檔案負責產生「假的但合理」半導體製程資料，讓其他範例程式可以
不用先下載資料集，也能練習良率預測、分群、視覺化與異常偵測。

情境說明：
假設我們是晶圓廠工程師，想知道溫度、壓力、製程時間與機台年限是否會
影響產品良率。這裡用規則建立 yield 欄位：

- yield = 1：良品
- yield = 0：不良品
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["temperature", "pressure", "time", "machine_age"]
TARGET_COLUMN = "yield"
RANDOM_STATE = 42


def create_process_data(
    n_samples: int = 1000,
    random_state: int = RANDOM_STATE,
    add_noise: bool = False,
    noise_rate: float = 0.10,
) -> pd.DataFrame:
    """建立一份模擬半導體製程資料。

    n_samples:
        要產生幾筆資料。
    random_state:
        固定亂數種子，讓每次執行的結果可重現。
    add_noise:
        是否加入隨機錯誤。現實製程不會完全照規則走，所以加入 noise
        可以讓資料更接近真實狀況。
    noise_rate:
        隨機翻轉 yield 的比例。
    """

    # 建立亂數產生器。固定 random_state 之後，教材示範時每次結果一致。
    rng = np.random.default_rng(random_state)

    # Step 1：建立製程參數。
    # temperature：製程溫度，假設平均約 200 度。
    # pressure：製程壓力，假設平均約 5。
    # time：製程時間，假設平均約 60 分鐘。
    # machine_age：機台年限，1 到 9 年。
    data = pd.DataFrame(
        {
            "temperature": rng.normal(200, 10, n_samples),
            "pressure": rng.normal(5, 1, n_samples),
            "time": rng.normal(60, 5, n_samples),
            "machine_age": rng.integers(1, 10, n_samples),
        }
    )

    # Step 2：建立良率規則。
    # 這裡假設溫度不要太高、壓力不要太高、機台不要太舊時，產品較可能是良品。
    target = (
        (data["temperature"] < 210)
        & (data["pressure"] < 6)
        & (data["machine_age"] < 7)
    ).astype(int)

    # Step 3：加入製程雜訊。
    # 真實工廠資料通常會有量測誤差、材料差異或未知因素，所以少量翻轉 label。
    if add_noise:
        if not 0 <= noise_rate <= 1:
            raise ValueError("noise_rate must be between 0 and 1.")
        noise_mask = rng.random(n_samples) < noise_rate
        target = target.copy()
        target.loc[noise_mask] = 1 - target.loc[noise_mask]

    # Step 4：把良率欄位加回資料表，讓其他程式可以直接使用。
    data[TARGET_COLUMN] = target
    return data


def split_features_target(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """把資料拆成特徵 X 與答案 y。

    X 是模型輸入，也就是製程參數。
    y 是模型要預測的結果，也就是良率 yield。
    """
    return data[FEATURE_COLUMNS], data[TARGET_COLUMN]
