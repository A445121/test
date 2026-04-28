"""使用 TensorFlow 神經網路預測半導體良率。

教材用途：
這是深度學習版的良率預測。和 Logistic Regression 一樣是二元分類，但神經
網路可以學習較複雜、非線性的製程關係。

學習重點：
- 為什麼神經網路需要特徵標準化
- Dense layer 如何組成簡單分類模型
- sigmoid 輸出如何轉成良品/不良品
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from simulated_data import RANDOM_STATE, create_process_data, split_features_target


def main() -> None:
    # TensorFlow 不是所有環境都會預裝，所以放在 main 裡檢查。
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is required. Install dependencies first.") from exc

    # 固定亂數，讓每次訓練結果比較容易重現。
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Step 1：建立模擬資料並拆成 X / y。
    data = create_process_data(add_noise=True)
    X, y = split_features_target(data)

    # Step 2：切分訓練集與測試集。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Step 3：標準化特徵。
    # 神經網路對數值尺度敏感，所以先把不同單位的欄位轉成平均約 0、標準差約 1。
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4：建立神經網路。
    # Dense(16) 和 Dense(8) 是隱藏層；最後 Dense(1)+sigmoid 輸出良品機率。
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Step 5：編譯與訓練模型。
    # binary_crossentropy 適合 0/1 二元分類問題。
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=1)

    # Step 6：評估模型。
    # predict 輸出的是 0 到 1 的機率，>= 0.5 就判斷為良品。
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    y_prob = model.predict(X_test_scaled, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print("TensorFlow loss:", loss)
    print("TensorFlow accuracy:", accuracy)
    print("Sklearn accuracy:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
