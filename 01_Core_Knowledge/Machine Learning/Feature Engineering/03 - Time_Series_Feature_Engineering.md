---
title: "Time Series Feature Engineering"
tags: [feature-engineering, time-series, datetime]
---
# Time Series Feature Engineering

## 1. Datetime features

Từ timestamp có thể tạo:
* Năm, quý, tháng
* Ngày, thứ
* Giờ, phút
* Cuối tuần
* Ngày lễ
* Mùa

---

## 2. Cyclical encoding

Giờ 23 và giờ 0 rất gần nhau về bản chất thời gian nhưng giá trị số thô lại cách nhau 23 đơn vị. Giải pháp là chuyển đổi qua hàm Sin/Cos:

$$x_{\sin} = \sin\left(2\pi\frac{x}{P}\right)$$

$$x_{\cos} = \cos\left(2\pi\frac{x}{P}\right)$$

**Ví dụ chu kỳ ($P$):**
* **Giờ:** $P = 24$
* **Tháng:** $P = 12$
* **Thứ:** $P = 7$

---

## 3. Lag features

$$Lag_k(t) = y_{t-k}$$

**Ví dụ các độ trễ phổ biến:**
* `lag_1` (1 bước thời gian trước)
* `lag_24` (24 giờ trước)
* `lag_168` (1 tuần trước)

---

## 4. Rolling features

$$\text{RollingMean}_w(t) = \frac{1}{w} \sum_{k=1}^{w} y_{t-k}$$

**Các thống kê cửa sổ trượt có thể tạo:**
* Rolling Mean
* Median
* Min / Max
* Standard Deviation
* Sum

> [!danger] Nguy cơ Data Leakage
> Phải **chỉ dùng dữ liệu trong quá khứ**. Thường cần thực hiện `.shift(1)` trước khi áp dụng hàm `.rolling()` để tránh rò rỉ dữ liệu của thời điểm hiện tại.

**Cách triển khai chuẩn trong pandas:**

```python
df["lag_1"] = df["target"].shift(1)

df["rolling_mean_7"] = (
    df["target"]
    .shift(1)
    .rolling(7)
    .mean()
)
```

---

## 5. Difference và Growth Rate

**Chênh lệch tuyệt đối (Difference):**
$$\Delta y_t = y_t - y_{t-1}$$

**Tốc độ tăng trưởng (Growth Rate):**
$$\text{Growth Rate} = \frac{y_t - y_{t-1}}{|y_{t-1}| + \epsilon}$$

---

## 6. External features (Biến ngoại sinh)

* Weather (Thời tiết)
* Holiday (Ngày lễ)
* Promotion (Khuyến mãi)
* Price (Giá cả)
* Traffic (Lưu lượng giao thông)
* Events (Sự kiện đặc biệt)
* Policy changes (Thay đổi chính sách)

---

## 7. Time-based validation

**Bắt buộc:** Không random shuffle (xáo trộn ngẫu nhiên) dữ liệu chuỗi thời gian.

$$\text{Past} \rightarrow \text{Train} \quad \Big| \quad \text{Later} \rightarrow \text{Validation} \quad \Big| \quad \text{Future} \rightarrow \text{Test}$$

---

## 8. Feature availability (Tính khả dụng của đặc trưng)

Feature ở tập train phải mô phỏng chính xác dữ liệu thực sự có mặt tại thời điểm dự báo trong thực tế (Inference Time).

* *Ví dụ:* Nếu dự báo $7$ ngày tới, **không được dùng thời tiết thực tế** của $7$ ngày tới; chỉ được dùng **dự báo thời tiết (weather forecast)** khả dụng đúng tại thời điểm chạy model.

---
