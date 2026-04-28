"""使用 AutoEncoder 做模擬製程異常偵測。

教材用途：
AutoEncoder 會學習「正常資料」長什麼樣子。當某筆資料重建誤差很大時，
代表它不像正常製程，因此可以視為異常。

學習重點：
- 只用正常資料訓練 AutoEncoder
- 用重建誤差判斷異常
- threshold 是異常判斷門檻
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from simulated_data import RANDOM_STATE, create_process_data, split_features_target


def main() -> None:
    # TensorFlow 不是所有環境都會預裝，所以放在 main 裡檢查。
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.models import Model
    except ImportError as exc:
        raise SystemExit("TensorFlow is required. Install dependencies first.") from exc

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Step 1：建立資料並拆出特徵與良率。
    data = create_process_data(add_noise=True)
    X, y = split_features_target(data)

    # Step 2：標準化特徵。
    # AutoEncoder 用數值距離計算重建誤差，所以不同欄位需要先轉到相近尺度。
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3：只取正常資料，也就是 yield=1 的良品製程。
    X_normal = X_scaled[y.to_numpy() == 1]

    # Step 4：建立 AutoEncoder。
    # encoded 是壓縮後的表示；decoded 是模型嘗試還原回原始特徵。
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(8, activation="relu")(input_layer)
    decoded = Dense(input_dim, activation="linear")(encoded)

    # Step 5：訓練模型學會重建正常資料。
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X_normal, X_normal, epochs=20, batch_size=32, verbose=1)

    # Step 6：計算每筆資料的重建誤差。
    # 誤差越大，代表資料越不像模型學到的正常模式。
    recon = autoencoder.predict(X_scaled, verbose=0)
    error = np.mean(np.square(X_scaled - recon), axis=1)

    # Step 7：設定異常門檻。
    # 這裡用平均誤差 + 2 倍標準差作為簡單門檻。
    threshold = error.mean() + 2 * error.std()
    anomaly = error > threshold

    print("Threshold:", threshold)
    print("Anomaly count:", int(anomaly.sum()))


if __name__ == "__main__":
    main()
