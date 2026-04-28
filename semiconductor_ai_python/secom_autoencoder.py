"""使用 AutoEncoder 對 SECOM 資料做異常偵測。

教材用途：
Word 文件中提到 AutoEncoder 可用於找異常晶圓、設備問題或壞製程。
這個檔案使用 SECOM 的良品資料訓練 AutoEncoder，再用重建誤差判斷測試資料
是否異常。

學習重點：
- label = 1 的良品資料可視為正常樣本
- AutoEncoder 學正常模式，不直接學分類規則
- 重建誤差越大，越可能是異常製程
"""

from __future__ import annotations

import argparse

import numpy as np

from secom_utils import RANDOM_STATE, preprocess_secom, split_and_scale


def main() -> None:
    # 讓使用者可以指定 SECOM 檔案位置。
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="secom.data")
    parser.add_argument("--labels-path", default="secom_labels.data")
    args = parser.parse_args()

    # TensorFlow 不是所有環境都會預裝，所以放在 main 裡檢查。
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.models import Model
    except ImportError as exc:
        raise SystemExit("TensorFlow is required. Install dependencies first.") from exc

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Step 1：讀取、清理、切分並標準化資料。
    X, y = preprocess_secom(args.data_path, args.labels_path)
    X_train, X_test, y_train, _, _ = split_and_scale(X, y)

    # Step 2：只取正常資料訓練，也就是 SECOM 中 label=1 的良品。
    X_normal = X_train[y_train == 1]
    input_dim = X_train.shape[1]

    # Step 3：建立 AutoEncoder。
    # 中間層會壓縮資料，迫使模型學到正常製程的主要模式。
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation="relu")(input_layer)
    encoded = Dense(16, activation="relu")(encoded)
    decoded = Dense(32, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    # Step 4：訓練模型重建正常資料。
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X_normal, X_normal, epochs=20, batch_size=32, verbose=1)

    # Step 5：用正常資料的重建誤差建立異常門檻。
    train_recon = autoencoder.predict(X_normal, verbose=0)
    train_error = np.mean(np.square(X_normal - train_recon), axis=1)
    threshold = train_error.mean() + 2 * train_error.std()

    # Step 6：計算測試資料重建誤差，超過門檻就判定為異常。
    test_recon = autoencoder.predict(X_test, verbose=0)
    test_error = np.mean(np.square(X_test - test_recon), axis=1)
    anomaly = test_error > threshold

    print("Threshold:", threshold)
    print("Test anomaly count:", int(anomaly.sum()))


if __name__ == "__main__":
    main()
