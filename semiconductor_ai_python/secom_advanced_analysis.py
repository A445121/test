"""SECOM 資料的進階分析：ROC、PCA 與 LSTM。

教材用途：
這個檔案整理 Word 文件最後提到的進階主題：

- ROC / AUC：評估分類模型能力
- PCA：把高維度 SECOM 特徵壓縮成 2 維，方便視覺化
- LSTM：把某個製程特徵當時間序列，示範設備預測維護概念

預設會執行 ROC 與 PCA；如果加上 --run-lstm，才會額外執行 LSTM。
"""

from __future__ import annotations

import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler

from secom_utils import RANDOM_STATE, preprocess_secom, split_and_scale


def create_dataset(values: np.ndarray, step: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """把連續數值切成「前 step 筆預測下一筆」的時間序列資料。"""
    X, y = [], []
    for i in range(len(values) - step):
        X.append(values[i : i + step])
        y.append(values[i + step])
    return np.array(X), np.array(y)


def run_lstm(series: np.ndarray) -> None:
    """使用 LSTM 示範時間序列預測。"""

    # TensorFlow 不是所有環境都會預裝，所以只有指定 --run-lstm 時才檢查。
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for --run-lstm.") from exc

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Step 1：標準化序列。
    series = (series - series.mean()) / series.std()

    # Step 2：整理成 LSTM 需要的三維格式。
    X_lstm, y_lstm = create_dataset(series, step=10)
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

    # Step 3：建立並訓練 LSTM 模型。
    model = Sequential(
        [
            LSTM(50, activation="relu", input_shape=(10, 1)),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=1)
    print("LSTM final loss:", history.history["loss"][-1])


def main() -> None:
    # 讓使用者可以指定 SECOM 檔案位置、是否執行 LSTM、是否存圖。
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="secom.data")
    parser.add_argument("--labels-path", default="secom_labels.data")
    parser.add_argument("--run-lstm", action="store_true")
    parser.add_argument("--save", help="Optional output path for the ROC/PCA figure.")
    args = parser.parse_args()

    # Step 1：讀取、清理、切分並標準化資料。
    X, y = preprocess_secom(args.data_path, args.labels_path)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)

    # Step 2：訓練 Random Forest。
    # Word 文件中提到 Random Forest 適合處理高維度、非線性資料。
    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    rf_model.fit(X_train, y_train)

    # Step 3：取得預測為良品 label=1 的機率，用來畫 ROC 曲線。
    positive_index = int(np.where(rf_model.classes_ == 1)[0][0])
    y_prob = rf_model.predict_proba(X_test)[:, positive_index]

    # Step 4：計算 ROC 與 AUC。
    # AUC 越接近 1，代表模型越能區分良品與不良品。
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Step 5：PCA 降到 2 維。
    # SECOM 原始特徵非常多，PCA 可協助視覺化資料分布。
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Step 6：畫出 ROC 曲線與 PCA 分布圖。
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", s=12)
    axes[1].set_title("PCA Visualization")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.colorbar(scatter, ax=axes[1], label="Label")

    fig.tight_layout()

    # Step 7：如果有指定 --save 就存圖，否則直接顯示。
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved advanced analysis figure to {args.save}")
    else:
        plt.show()

    print("Random Forest ROC AUC:", roc_auc)

    # Step 8：選配 LSTM 範例。
    # 因為 LSTM 需要 TensorFlow，且訓練較久，所以預設不執行。
    if args.run_lstm:
        first_feature = X.iloc[:, 0].to_numpy(dtype=float)
        run_lstm(first_feature)


if __name__ == "__main__":
    main()
