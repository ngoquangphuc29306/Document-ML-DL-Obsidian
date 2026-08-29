---
tags:
  - feature-engineering
  - evaluation
  - time-series
---

# Chuyển Đổi Time Series Sang Supervised Learning & Time Series Cross-Validation

## 1. Bản Chất Vấn Đề (Motivation)
Hầu hết các thuật toán Học máy có giám sát (Supervised Learning) như *Random Forest, XGBoost, Support Vector Machines (SVM)* đều thiết lập trên giả định dữ liệu dạng bảng $(X, y)$ với các mẫu độc lập và phân phối đồng nhất (i.i.d - *independent and identically distributed*).

Tuy nhiên, dữ liệu **Chuỗi thời gian (Time Series)** lại mang tính chất tự tương quan (Autocorrelation) — giá trị ở thời điểm $t$ phụ thuộc chặt chẽ vào các thời điểm trong quá khứ ($t-1, t-2, \dots$). 

> **Mục tiêu:** Sử dụng kỹ thuật Tiền xử lý dữ liệu để biến đổi chuỗi giá trị theo thời gian $S = [x_1, x_2, x_3, \dots, x_N]$ thành tập dữ liệu có giám sát dạng cặp $(X_{input}, y_{target})$.

---

## 2. Kỹ Thuật Chuyển Đổi (Time Series to Supervised Learning)

### 2.1 Phương Pháp Cửa Sổ Trượt (Sliding Window / Lag Features)
Phương pháp này dùng $K$ giá trị ở các bước thời gian quá khứ (gọi là **Lags**) để làm đặc trưng đầu vào ($X$) nhằm dự đoán giá trị ở bước tiếp theo ($y_t$).

#### Biểu diễn Toán học:
Dự đoán giá trị $\hat{y}_t$ tại thời điểm $t$ thông qua $K$ bước trễ:

$$\hat{y}_t = f(x_{t-1}, x_{t-2}, \dots, x_{t-K})$$

* $y_t = x_t$: Giá trị mục tiêu (Target) tại thời điểm $t$.
* $x_{t-k}$: Đặc trưng độ trễ thứ $k$ (Lag $k$).
* $K$: Kích thước cửa sổ quan sát (Window size).

#### Minh họa Biến đổi Dữ liệu:
![[Pasted image 20260723222634.png]]
#### Code Python
```python
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
```

### Bảng So Sánh Các Kiểu Dự Đoán (Forecasting Types)

| Tiêu Chí Phân Loại | Tên Kiểu Dự Đoán | Định Nghĩa / Bản Chất | Ví Dụ Thực Tế | Đặc Điểm Dữ Liệu ($X \to y$) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Số Lượng Biến Đầu Vào** *(Input Features)* | **Univariate Forecasting** *(Dự đoán đơn biến)* | Chỉ sử dụng **1 chuỗi dữ liệu trong quá khứ** của chính biến đó để dự đoán tương lai. | Dự đoán giá cổ phiếu VIC dựa trên lịch sử giá cổ phiếu VIC. | $X = [y_{t-k}, \dots, y_{t-1}]$<br>$y = [y_t]$ |
| | **Multivariate Forecasting** *(Dự đoán đa biến)* | Sử dụng **nhiều chuỗi dữ liệu (nhiều biến)** có tương quan với nhau để dự đoán biến mục tiêu. | Dự đoán Giá nhà dựa trên Lãi suất, Lạm phát và Giá vật liệu. | $X = [x1, x2, x3]_{t-k \dots t-1}$<br>$y = [y_t]$ |
| **2. Tầm Nhìn Dự Đoán** *(Forecast Horizon)* | **Single-step Forecasting** *(Dự đoán 1 bước)* | Chỉ dự đoán **1 điểm thời gian tiếp theo** ngay sau tập huấn luyện ($t+1$). | Dự đoán nhiệt độ của 1 giờ tiếp theo. | $y = [y_{t+1}]$ |
| | **Multi-step Forecasting** *(Dự đoán nhiều bước)* | Dự đoán **một dải/một chuỗi thời gian** liên tiếp trong tương lai ($t+1, t+2, \dots, t+H$). | Dự đoán doanh số bán hàng cho cả 7 ngày tới trong tuần. | $y = [y_{t+1}, y_{t+2}, \dots, y_{t+H}]$ |
| **3. Số Lượng Mục Tiêu Đầu Ra** *(Output Targets)* | **Single-output Forecasting** | Chỉ đưa ra kết quả dự đoán cho **1 biến mục tiêu**. | Dự đoán chỉ số Nhiệt độ môi trường. | $y = [\text{Temp}_{t+1}]$ |
| | **Multi-output Forecasting** | Đưa ra kết quả dự đoán đồng thời cho **nhiều biến mục tiêu cùng lúc**. | Dự đoán đồng thời cả Nhiệt độ VÀ Độ ẩm cho ngày mai. | $y = [\text{Temp}_{t+1}, \text{Humidity}_{t+1}]$ |

---

###  Ánh Xạ Với Hàm `series_to_supervised`

Khi triển khai code với hàm `series_to_supervised(data, n_in, n_out)`:

* **`data`**:
  * Dữ liệu 1D / 1 cột $\to$ **Univariate**
  * Dữ liệu 2D / nhiều cột $\to$ **Multivariate**
* **`n_in`**: Số lượng Lag features ($K$ bước quá khứ làm $X$).
* **`n_out`**:
  * `n_out = 1` $\to$ **Single-step Forecasting**
  * `n_out > 1` $\to$ **Multi-step Forecasting**

---
## 3. Kiểm Thử Chuỗi Thời Gian (Time Series Cross-Validation)

### Tại Sao Không Dùng K-Fold Cross ([[Cross-Validation]])
-Validation Thông Thường?
Trong K-Fold Cross-Validation truyền thống, dữ liệu bị xáo trộn ngẫu nhiên (Random Shuffle). 

Nếu áp dụng cho Time Series:
* Dữ liệu ở tương lai ($t+1$) sẽ rơi vào tập **Train**, còn dữ liệu quá khứ ($t$) lại rơi vào tập **Validation**.
* Dẫn đến hiện tượng **Data Leakage (Rò rỉ dữ liệu tương lai)**, làm cho đánh giá mô hình bị lạc quan quá mức so với thực tế triển khai.

### Walk-Forward Validation chia làm 2 loại:

![[Pasted image 20260723222743.png]]