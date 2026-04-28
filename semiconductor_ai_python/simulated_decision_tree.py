"""使用 Decision Tree 找出影響良率的關鍵因素。

教材用途：
Decision Tree 的優點是容易解釋。它會像流程圖一樣，把資料依照條件切分，
例如 temperature < 某個門檻、pressure < 某個門檻，最後判斷良品或不良品。

學習重點：
- 決策樹如何學習分類規則
- 如何視覺化決策樹
- 如何從樹狀圖看出重要製程參數
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from simulated_data import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    create_process_data,
    split_features_target,
)


def main() -> None:
    # 讓使用者可以選擇把圖存成檔案。
    # 範例：python simulated_decision_tree.py --save tree.png
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Optional output path for the tree image.")
    args = parser.parse_args()

    # Step 1：建立模擬資料並拆成 X / y。
    data = create_process_data(add_noise=False)
    X, y = split_features_target(data)

    # Step 2：切分資料。
    # 這個檔案主要是畫決策樹，所以只需要訓練集來建立樹。
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Step 3：建立決策樹模型。
    # max_depth=3 讓樹不要太深，教材展示時比較容易閱讀。
    model_tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    model_tree.fit(X_train, y_train)

    # Step 4：把模型畫成樹狀圖。
    # filled=True 會用顏色標示分類傾向；feature_names 顯示各個製程欄位名稱。
    plt.figure(figsize=(12, 8))
    tree.plot_tree(
        model_tree,
        feature_names=FEATURE_COLUMNS,
        class_names=["bad", "good"],
        filled=True,
        rounded=True,
    )
    plt.title("Decision Tree for Yield Factors")

    # Step 5：如果有指定 --save 就存檔，沒有指定就直接顯示圖表。
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved decision tree to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
