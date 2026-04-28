"""使用 K-means 對半導體製程條件分群。

教材用途：
K-means 是非監督式學習。它不看 yield 答案，而是只根據製程參數，把相似的
製程條件分到同一群。

學習重點：
- 分群和分類的差異
- 如何找出相似製程條件
- 如何觀察每一群的平均良率
"""

from __future__ import annotations

from sklearn.cluster import KMeans

from simulated_data import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, create_process_data


def main() -> None:
    # Step 1：建立模擬資料。
    data = create_process_data(add_noise=True)

    # Step 2：K-means 只使用特徵欄位，不直接使用 yield 當答案。
    X = data[FEATURE_COLUMNS]

    # Step 3：建立 3 群製程條件。
    # n_init=10 代表模型會嘗試多次不同起點，再選出較好的分群結果。
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
    data["cluster"] = kmeans.fit_predict(X)

    # Step 4：查看每一群的平均製程條件與平均良率。
    # 如果某一群 yield 平均值較高，可視為比較好的製程群。
    summary_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    print(data.groupby("cluster")[summary_columns].mean().round(3))


if __name__ == "__main__":
    main()
