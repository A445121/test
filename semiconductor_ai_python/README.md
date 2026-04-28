# 半導體產業 AI Python 教材

本資料夾依照 `半導體產業 AI 規劃_python.docx` 內容整理而成，目標是把 Word
裡的程式碼變成有架構、可閱讀、可教學的 Python 範例。

## 情境

假設你是晶圓廠工程師，想用 AI 協助解決三個問題：

1. 良率預測：哪些晶圓可能是良品或不良品？
2. 設備異常偵測：哪些製程資料看起來不像正常狀態？
3. 製程最佳化：哪些參數會影響品質？

## 欄位說明

模擬資料使用下列欄位：

- `temperature`：製程溫度
- `pressure`：製程壓力
- `time`：製程時間
- `machine_age`：機台年限
- `yield`：良率結果，`1` 代表良品，`0` 代表不良品

SECOM 真實資料使用：

- `feature_0`, `feature_1`, ...：匿名製程參數
- `label`：`1` 代表良品，`-1` 代表不良品

## 程式檔案對照

### 模擬資料版

- `simulated_data.py`：產生模擬半導體製程資料
- `simulated_yield_prediction.py`：Logistic Regression 良率預測
- `simulated_decision_tree.py`：Decision Tree 找關鍵製程因素
- `simulated_kmeans.py`：K-means 製程條件分群
- `simulated_eda.py`：良率分布與溫度分布視覺化
- `simulated_deep_learning.py`：TensorFlow 神經網路良率預測
- `simulated_lstm.py`：LSTM 製程時間序列預測
- `simulated_autoencoder.py`：AutoEncoder 異常偵測

### SECOM 真實資料版

- `secom_utils.py`：SECOM 讀檔、缺失值處理、切分資料、標準化
- `secom_eda.py`：SECOM EDA 圖表
- `secom_model_comparison.py`：Logistic Regression、SVM、Random Forest 比較
- `secom_autoencoder.py`：SECOM AutoEncoder 異常偵測
- `secom_advanced_analysis.py`：ROC、AUC、PCA、LSTM 進階分析

## 執行方式

安裝套件：

```bash
pip install -r requirements.txt
```

執行模擬資料版本：

```bash
python simulated_yield_prediction.py
python simulated_eda.py
python simulated_decision_tree.py
```

SECOM 版本需要先放入兩個資料檔：

- `secom.data`
- `secom_labels.data`

執行 SECOM 版本：

```bash
python secom_eda.py
python secom_model_comparison.py
python secom_advanced_analysis.py
```

## QA

### Q1：這份專題在做什麼？

它是在示範 AI 如何應用於半導體製程，包含良率預測、製程分群、異常偵測與
模型評估。

### Q2：為什麼要預測良率？

半導體製程複雜，如果能提前預測哪些晶圓可能不良，就能協助工程師調整製程、
降低成本並提升生產效率。

### Q3：`yield` 是什麼？

在模擬資料中，`yield = 1` 代表良品，`yield = 0` 代表不良品。SECOM 資料則是
`label = 1` 代表良品，`label = -1` 代表不良品。

### Q4：為什麼使用 `temperature`、`pressure`、`time`、`machine_age`？

這些欄位對應 Word 內容中的半導體製程變數。溫度、壓力與時間會影響材料反應，
機台年限則可能影響設備穩定性。

### Q5：Logistic Regression、Decision Tree、K-means 差在哪？

Logistic Regression 用來預測良品或不良品；Decision Tree 用來找出判斷良率的
關鍵條件；K-means 不看答案，而是把相似的製程條件分成群組。

### Q6：為什麼要加入 noise？

真實製程資料不會完全照規則產生，可能有量測誤差、材料差異或未知因素。加入
noise 可以讓模擬資料更接近真實情境。

### Q7：SECOM 資料為什麼需要缺失值處理？

SECOM 是真實半導體製程資料，含有大量缺失值。若不先移除缺失太多的欄位並填補
剩餘缺失值，模型通常無法直接訓練。

### Q8：為什麼 Random Forest 可能表現較好？

SECOM 屬於高維度且可能非線性的資料。Random Forest 由多棵決策樹組成，通常比
單純線性模型更能捕捉複雜特徵關係。

### Q9：AutoEncoder 為什麼可以做異常偵測？

AutoEncoder 先學習正常資料的樣子。當某筆資料重建誤差很大，代表它不像正常製程，
因此可被視為異常樣本。

### Q10：ROC / AUC 在看什麼？

ROC 曲線用來評估模型區分良品與不良品的能力。AUC 越接近 1，代表模型分類能力
越好。

### Q11：PCA 的用途是什麼？

SECOM 有很多特徵，PCA 可以把高維度資料壓縮成較低維度，方便視覺化，也可用來
理解資料是否有分群或分離趨勢。

### Q12：LSTM 為什麼適合設備預測？

半導體設備資料常隨時間變化。LSTM 適合處理時間序列，可用於預測設備狀態或製程
漂移。
