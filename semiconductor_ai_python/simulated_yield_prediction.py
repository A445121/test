"""使用 Logistic Regression 預測半導體良率。

教材用途：
這是最基礎的監督式學習範例。程式會先產生模擬製程資料，再用
temperature、pressure、time、machine_age 預測 yield。

學習重點：
- 如何產生訓練資料
- 如何把資料切成訓練集與測試集
- 如何訓練 Logistic Regression
- 如何用 Accuracy 與 classification report 解讀模型表現
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from simulated_data import RANDOM_STATE, create_process_data, split_features_target


def main() -> None:
    # Step 1：建立模擬資料。
    # add_noise=True 代表加入少量隨機錯誤，讓資料不要太完美。
    data = create_process_data(add_noise=True)

    # Step 2：拆出模型輸入 X 與答案 y。
    # X：製程參數；y：良率結果。
    X, y = split_features_target(data)

    # Step 3：切分訓練集與測試集。
    # 訓練集用來讓模型學習；測試集用來檢查模型是否真的能預測新資料。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Step 4：建立並訓練 Logistic Regression 模型。
    # Logistic Regression 常用於二元分類，例如良品/不良品。
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Step 5：使用測試集做預測。
    y_pred = model.predict(X_test)

    # Step 6：輸出資料樣貌與模型結果。
    # Accuracy 越高，代表預測正確比例越高；classification report 可看各類別表現。
    print(data.head())
    print()
    print("Yield distribution:")
    print(y.value_counts().sort_index())
    print()
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
