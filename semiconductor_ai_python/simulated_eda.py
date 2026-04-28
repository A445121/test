"""模擬半導體製程資料的 EDA 視覺化。

教材用途：
EDA 是 Exploratory Data Analysis，意思是建模前先探索資料。這個檔案用圖表
觀察良品/不良品數量，以及溫度分布。

學習重點：
- 看良率分布是否平衡
- 比較不同 yield 的平均製程參數
- 用圖表快速理解資料特性
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from simulated_data import TARGET_COLUMN, create_process_data


def main() -> None:
    # 讓使用者可以把 EDA 圖存成圖片。
    # 範例：python simulated_eda.py --save eda.png
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Optional output path for the EDA figure.")
    args = parser.parse_args()

    # Step 1：建立模擬資料。
    data = create_process_data(add_noise=True)

    # Step 2：輸出良品/不良品數量。
    # 這可以看資料是否有類別不平衡問題。
    print("Yield counts:")
    print(data[TARGET_COLUMN].value_counts().sort_index())
    print()

    # Step 3：依照 yield 分組，看良品與不良品的平均製程條件差異。
    print("Mean values by yield:")
    print(data.groupby(TARGET_COLUMN).mean(numeric_only=True).round(3))

    # Step 4：建立兩張圖。
    # 左圖是良率分布；右圖是溫度分布。
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    data[TARGET_COLUMN].value_counts().sort_index().plot(kind="bar", ax=axes[0])
    axes[0].set_title("Yield Distribution")
    axes[0].set_xlabel("Yield")
    axes[0].set_ylabel("Count")

    axes[1].hist(data["temperature"], bins=30)
    axes[1].set_title("Temperature Distribution")
    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("Count")

    fig.tight_layout()

    # Step 5：如果有指定 --save 就存檔，否則直接顯示。
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved EDA figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
