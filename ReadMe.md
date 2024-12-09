# LibriSpeech Transformer Training

## 項目簡介
本項目使用 PyTorch 和 Transformer 模型對 LibriSpeech 數據集進行訓練，通過提取音頻的 Mel 頻譜特徵並與文本標註對應來構建語音處理模型。

## 目錄結構
```
project/
│
├── models.py       # 包含 SimpleTransformer 模型
├── dataset.py      # 包含 LibriSpeechDataset 和 collate_fn
├── utils.py        # 包含 extract_mel_spectrogram 工具函數
└── train.py        # 主執行文件，進行訓練和測試
```

## 文件功能
- **models.py**  
  包含 `SimpleTransformer` 模型，基於 Transformer 結構處理音頻數據。

- **dataset.py**  
  提供 `LibriSpeechDataset` 類，用於加載 LibriSpeech 數據集，提取音頻特徵並匹配文本標註。

- **utils.py**  
  提供 `extract_mel_spectrogram` 函數，用於處理音頻特徵提取。

- **train.py**  
  主執行文件，用於訓練模型。

---

## 執行順序

1. **準備數據和依賴**：下載 LibriSpeech 數據集，並安裝所需依賴庫。
2. **修改參數**（可選）：根據需求修改 `models.py` 或 `train.py` 中的參數。
3. **執行 `train.py`**：啟動訓練過程。
   ```bash
   python train.py
   ```
4. **驗證模型輸出**（可選）：在 `train.py` 中添加測試代碼，進行模型結果檢查。

---

## 聯繫方式
如有問題，請聯繫：`your-email@example.com`
