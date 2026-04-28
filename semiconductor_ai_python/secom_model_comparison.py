"""比較 SECOM 資料上的三種機器學習模型。

教材用途：
Word 文件中提到要做 Logistic Regression、SVM、Random Forest 模型比較。
這個檔案會用相同資料切分方式訓練三個模型，並輸出 Accuracy 與分類報告。

學習重點：
- 不同模型如何處理同一份高維度資料
- class_weight="balanced" 如何處理良品/不良品比例不平衡
- 為什麼不能只看 Accuracy，還要看 precision、recall、f1-score
"""

from __future__ import annotations

import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

from secom_utils import RANDOM_STATE, preprocess_secom, split_and_scale


def main() -> None:
    # 讓使用者可以指定 SECOM 檔案位置。
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="secom.data")
    parser.add_argument("--labels-path", default="secom_labels.data")
    args = parser.parse_args()

    # Step 1：讀取、清理、切分並標準化資料。
    X, y = preprocess_secom(args.data_path, args.labels_path)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)

    # Step 2：建立要比較的模型。
    # Logistic Regression：基礎線性分類模型。
    # SVM：可處理高維度資料的分類模型。
    # Random Forest：多棵決策樹組成，能捕捉非線性關係。
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    # Step 3：逐一訓練、預測、輸出評估結果。
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n{name}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
