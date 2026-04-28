"""SECOM 真實半導體資料的 EDA 分析。

教材用途：
這個檔案對 SECOM 資料做探索性資料分析，對應 Word 文件中的 EDA 圖表：

- 良率分布圖
- 單一特徵分布
- 前 20 個特徵的相關性熱力圖
- 良品與不良品在某個特徵上的分布比較
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from secom_utils import LABEL_COLUMN, preprocess_secom


def main() -> None:
    # 讓使用者可以指定 SECOM 檔案位置，也可以選擇把圖存成圖片。
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="secom.data")
    parser.add_argument("--labels-path", default="secom_labels.data")
    parser.add_argument("--save", help="Optional output path for the EDA figure.")
    args = parser.parse_args()

    # Step 1：讀取並清理 SECOM 資料。
    X, y = preprocess_secom(args.data_path, args.labels_path)

    # Step 2：把 X 和 y 合併成同一張表，方便分組與畫圖。
    data = X.copy()
    data[LABEL_COLUMN] = y

    # Step 3：選出要示範的欄位。
    # SECOM 欄位很多，教材中只取第一個特徵與前 20 個特徵做示範。
    first_feature = X.columns[0]
    subset = X.iloc[:, :20]
    good = data[data[LABEL_COLUMN] == 1]
    bad = data[data[LABEL_COLUMN] == -1]

    # Step 4：輸出資料形狀與良品/不良品數量。
    # 這可以讓我們先確認資料是否有類別不平衡。
    print("Preprocessed shape:", data.shape)
    print("Label counts:")
    print(y.value_counts().sort_index())

    # Step 5：建立四張 EDA 圖。
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 圖 1：良率分布。
    y.value_counts().sort_index().plot(kind="bar", ax=axes[0, 0])
    axes[0, 0].set_title("Yield Distribution")
    axes[0, 0].set_xlabel("Class (-1: Fail, 1: Pass)")
    axes[0, 0].set_ylabel("Count")

    # 圖 2：單一特徵分布。
    axes[0, 1].hist(data[first_feature], bins=50)
    axes[0, 1].set_title(f"{first_feature} Distribution")

    # 圖 3：相關性熱力圖。
    # 如果兩個特徵高度相關，代表可能有重複資訊，後續可考慮降維。
    sns.heatmap(subset.corr(), cmap="coolwarm", ax=axes[1, 0])
    axes[1, 0].set_title("Feature Correlation Heatmap")

    # 圖 4：良品與不良品在同一特徵上的比較。
    axes[1, 1].hist(good[first_feature], alpha=0.5, label="Good")
    axes[1, 1].hist(bad[first_feature], alpha=0.5, label="Bad")
    axes[1, 1].legend()
    axes[1, 1].set_title(f"{first_feature} Good vs Bad")

    fig.tight_layout()

    # Step 6：如果有指定 --save 就存檔，否則直接顯示。
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved EDA figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
