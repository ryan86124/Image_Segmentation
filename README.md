# Animal Metrology System

## 專案介紹
本專案利用 YOLOv8-Seg 實作影像分割，針對 COCO 資料集中的動物進行個體識別與精密量測。

## 技術
- **AI 模型**: YOLOv8-Seg (Instance Segmentation)
- **影像處理**: OpenCV, NumPy[cite: 1]
- **資料分析**: Pandas[cite: 1]

## 量測公式與驗證
- **距離公式**: 採用歐幾里得距離 $d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$[cite: 1, 3]
- **評估標準**: 使用 mAP (mean Average Precision) 評估分割精確度[cite: 1]
- **驗證方式**: 比對人工標註與系統量測值之 RMSE (均方根誤差)[cite: 1]

## 運行步驟
1. `pip install -r requirements.txt`
2. 設定 `.env` 檔案
3. `python main.py`

## Docker 部署
`docker build -t metrology-app .`
`docker run metrology-app`