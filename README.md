# Animal Metrology System

## 1. 專案核心介紹
本專案為一套自動化影像量測系統，利用深度學習技術對影像中的動物進行精確識別、實例分割（Instance Segmentation），並計算動物特徵點之間的像素距離。本專案已完成從環境建置 (Docker)、模型推理 (YOLOv8) 到數據結果輸出 (CSV) 的完整自動化流程。

---

## 2. AI 模型選擇與評估

### 使用模型：YOLOv8-seg (Instance Segmentation)
本系統採用 **YOLOv8-seg** 作為核心引擎，主要原因如下：
- **像素級精確度**：不同於傳統 Bounding Box 偵測，實例分割能提取精確的 Mask，對於定位動物邊緣與模擬特徵點（e.g.眼部座標）至關重要。
- **性能平衡**：在保持高推理速度的同時，具備極佳的偵測準確度，適合雲端伺服器環境大規模部署。
- **類別通用性**：原生支援 COCO 資料集中多種動物類別（如羊、狗、貓、牛等），具備良好的泛化能力。

### 評估標準：mAP (mean Average Precision)
系統效能採用 **mAP** 進行評估，其公式如下：

$$\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i$$

其中 $N$ 為類別總數，$AP_i$ 為單一類別在不同 Recall 下的平均精確度。這能確保系統在不同光照與角度下皆能穩定識別動物。

---

## 3. 量測方式與驗證邏輯

### 量測公式：歐幾里得距離 (Euclidean Distance)
系統透過分析分割遮罩（Mask），提取特徵點（如利用 Mask 橫向極值模擬眼部座標）進行量測。任意兩點 $(x_1, y_1)$ 與 $(x_2, y_2)$ 之間的距離 $d$ 採用歐幾里得公式計算：

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

- **單體量測 (Intra-individual)**：計算同一個體內兩特徵點（如左眼與右眼）的間距。
- **個體間量測 (Inter-individual)**：計算不同個體對應特徵點之間的間距（例如個體 A 右眼至個體 B 右眼）。

### 驗證方式：RMSE (均方根誤差)
為確保系統量測值之可靠性，可透過 RMSE 比對系統量測值 ($\hat{y}$) 與人工標註真值 ($y$)：

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

透過最小化 RMSE，能有效評估並校正影像量測中的系統誤差。

---

## 4. 系統架構圖
本專案採用 **Docker 容器化** 架構，確保開發環境與生產環境的一致性，簡化部署流程。

```mermaid
graph LR
    subgraph "Local environment (Windows/WSL)"
        Input_Image[Input: animals.jpg]
    end

    subgraph "Docker Container (Ubuntu/Python3)"
        direction TB
        YOLO_Model[YOLOv8-seg Engine]
        Pre_Process[Image Pre-processing]
        Post_Process[Metrology Calculation]
    end

    subgraph "Output Results"
        CSV[Data Report: metrology_results.csv]
        Visual[Annotated Image: result.jpg]
    end

    Input_Image -->|Volume Mount| Pre_Process
    Pre_Process --> YOLO_Model
    YOLO_Model --> Post_Process
    Post_Process -->|Euclidean Computation| CSV
    Post_Process --> Visual