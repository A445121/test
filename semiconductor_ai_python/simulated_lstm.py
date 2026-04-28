"""使用 LSTM 做製程溫度的時間序列預測。

教材用途：
Word 文件中提到 LSTM 可用於設備預測維護與製程漂移監控。這個範例把
temperature 當成一條時間序列，學習用前 10 筆溫度預測下一筆溫度。

學習重點：
- 如何把一維序列整理成 LSTM 需要的三維資料
- LSTM 適合處理時間順序資料
- loss 越低代表預測誤差越小
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from simulated_data import RANDOM_STATE, create_process_data


def create_dataset(values: np.ndarray, time_step: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """把連續數值切成「前 time_step 筆預測下一筆」的訓練資料。"""
    X, y = [], []
    for i in range(len(values) - time_step):
        X.append(values[i : i + time_step])
        y.append(values[i + time_step])
    return np.array(X), np.array(y)


def main() -> None:
    # TensorFlow 不是所有環境都會預裝，所以放在 main 裡檢查。
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential
    except ImportError as exc:
        raise SystemExit("TensorFlow is required. Install dependencies first.") from exc

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Step 1：建立模擬資料，並取出 temperature 當時間序列。
    data = create_process_data()
    series = data["temperature"].to_numpy(dtype=float)

    # Step 2：標準化序列，讓模型比較穩定。
    series = (series - series.mean()) / series.std()

    # Step 3：把序列轉成 LSTM 格式。
    # LSTM 輸入格式是 (樣本數, 時間步長, 特徵數)。
    X_lstm, y_lstm = create_dataset(series, time_step=10)
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

    # Step 4：建立 LSTM 模型。
    # LSTM(50) 負責學習時間依賴關係；Dense(1) 輸出下一個溫度值。
    model = Sequential(
        [
            LSTM(50, activation="relu", input_shape=(10, 1)),
            Dense(1),
        ]
    )

    # Step 5：訓練模型並輸出最後的訓練誤差。
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=1)
    print("Final loss:", history.history["loss"][-1])


if __name__ == "__main__":
    main()
