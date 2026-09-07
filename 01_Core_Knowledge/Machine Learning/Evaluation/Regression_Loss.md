---
tags: [machine-learning, deep-learning, regression, loss-functions, optimization, ai]
aliases: [Regression Loss Functions, Hàm mất mát hồi quy]
date: 2026-09-04
---

# Regression Loss Functions (Các hàm mất mát trong Hồi quy)

## 1. Tổng quan về Bài toán Hồi quy & Hàm Mất mát

### 1.1. Bản chất của Bài toán Hồi quy
Trong Machine Learning, bài toán hồi quy (Regression) tập trung vào việc mô hình hóa mối quan hệ ánh xạ giữa vector các biến độc lập (predictor variables) $x \in \mathbb{R}^D$ và một biến phụ thuộc liên tục (continuous outcome) $y \in \mathbb{R}$ thông qua mô hình tham số hóa $f_\Theta(x)$:

$$y = f_\Theta(x)$$

> [!NOTE] Cần ảnh minh họa: Đồ thị hồi quy trực quan thể hiện đường thẳng khớp dữ liệu (Regression Line), các điểm dữ liệu quan sát $(x_i, y_i)$, giá trị dự đoán $(x_i, \hat{y}_i)$, khoảng chênh lệch thẳng đứng (Vertical Offset / Residual $e_i = y_i - \hat{y}_i$), cùng hệ số góc $w_1$ và hệ số chặn $w_0$.

### 1.2. Vai trò của Hàm Mất mát (Loss Function)
Hàm mất mát $\mathcal{L}(\Theta)$ đóng vai trò là thước đo định lượng mức độ sai lệch giữa dự đoán của mô hình và giá trị thực tế trên tập dữ liệu gồm $N$ mẫu quan sát:

$$\min_\Theta \mathcal{L}(\Theta)$$

### 1.3. Cây Phân loại các Hàm Mất mát Hồi quy

```
                      ┌─────────────── Regression Loss ───────────────┐
                      │                                               │
               Mean Bias Error (MBE)                                  │
                      │                                               │
         ┌────────────┴────────────┐                                  │
         │                         │                                  │
Mean Absolute Error (MAE)    Mean Squared Error (MSE)           Log-Cosh Loss
         │                         │                          (Non-parametric)
         │                         ├────────────────────────┐
         │                         │                        │
   ┌─────┴─────┐         Root Mean Squared Error   Root Mean Squared Log
   │           │                 (RMSE)                 Error (RMSLE)
Huber      Smooth L1
 (Parametric / Piecewise)
```

> [!NOTE] Cần ảnh minh họa: Sơ đồ cây phân nhánh hệ thống các hàm Regression Loss (phân nhóm cơ bản, mở rộng từ MSE và các hàm dung hòa Parametric/Non-parametric).

---

## 2. Nhóm Hàm Loss Cơ bản

### 2.1. Mean Bias Error (MBE)
MBE đo lường sai số trung bình có dấu giữa giá trị thực tế và giá trị dự đoán, dùng để đánh giá độ thiên lệch hệ thống của mô hình.

* **Công thức toán học:**
  $$\mathcal{L}_{MBE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))$$

* **Phân tích dấu của sai số:**
  * $\mathcal{L}_{MBE} > 0$: Mô hình có xu hướng dự đoán **thấp hơn** thực tế (Underestimation).
  * $\mathcal{L}_{MBE} < 0$: Mô hình có xu hướng dự đoán **cao hơn** thực tế (Overestimation).
  * $\mathcal{L}_{MBE} \approx 0$: Mô hình không có độ lệch hệ thống rõ rệt.

> [!WARNING] Hiện tượng Triệt tiêu Sai số (Error Cancellation)
> Do giữ nguyên dấu, các sai số dương và âm có thể tự triệt tiêu lẫn nhau. Một mô hình có sai số dao động lớn ($\pm 100$) vẫn có thể cho $\text{MBE} \approx 0$. Do đó, **trong thực tế không ai sử dụng MBE làm hàm mất mát để huấn luyện tối ưu trực tiếp**, mà chỉ dùng như một chỉ số chẩn đoán để hiệu chỉnh mô hình (Calibration).

---

### 2.2. Mean Squared Error (MSE - Chuẩn $L_2$)
MSE tính trung bình bình phương khoảng cách giữa giá trị thực tế và giá trị dự đoán.

* **Công thức toán học:**
  $$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2$$

* **Đặc tính cốt lõi:**
  * **Toán học thuận lợi:** Là hàm lồi, liên tục và khả vi tại mọi điểm trên $\mathbb{R}$, cực kỳ tối ưu cho các thuật toán tính đạo hàm như Gradient Descent.
  * **Phạt nặng sai số lớn:** Nhờ phép bình phương, sai số càng lớn sẽ bị khuếch đại theo cấp số mũ.
  * **Điểm yếu chí mạng:** Cực kỳ nhạy cảm với các điểm dữ liệu ngoại lai (**Outliers**). Một vài mẫu ngoại lai có thể kéo lệch hoàn toàn đường hồi quy của mô hình.

> [!NOTE] Cần ảnh minh họa: Đồ thị hàm bậc hai dạng parabol đối xứng của MSE thể hiện độ dốc tăng mạnh khi sai số tăng cao.

---

### 2.3. Mean Absolute Error (MAE - Chuẩn $L_1$)
MAE đo lường giá trị tuyệt đối trung bình của các sai số, phản ánh khoảng cách độ lớn đơn thuần không xét chiều hướng.

* **Công thức toán học:**
  $$\mathcal{L}_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - f(x_i)|$$

* **Đặc tính cốt lõi:**
  * **Bền vững với Outliers (Robustness):** Sai số đóng góp theo tỷ lệ tuyến tính, không bị thổi phồng bởi lũy thừa.
  * **Đơn vị trực quan:** Giá trị loss giữ nguyên đơn vị gốc của biến mục tiêu, giúp việc giải thích kết quả dễ dàng.
  * **Hạn chế:** Không khả vi tại điểm sai số bằng $0$, gây khó khăn nhất định cho các thuật toán tối ưu dựa vào đạo hàm bậc nhất. Đồng thời, MAE đối xử với mọi mức độ sai số như nhau.

> [!NOTE] Cần ảnh minh họa: Đồ thị hàm chữ V của MAE thể hiện đạo hàm không liên tục tại gốc tọa độ $0$.

---

## 3. Nhóm Hàm Loss Mở rộng từ MSE

### 3.1. Root Mean Squared Error (RMSE)
RMSE là căn bậc hai của MSE, được đưa ra để khắc phục nhược điểm mất tính trực quan về mặt đơn vị của MSE.

* **Công thức toán học:**
  $$\mathcal{L}_{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2}$$

* **Đặc tính cốt lõi:**
  * Đưa đơn vị đo lường quay về đúng thang đo nguyên bản của biến mục tiêu $y$.
  * Kế thừa trọn vẹn đặc tính của MSE: nhạy cảm mạnh với ngoại lai và phạt nặng các sai lệch biên độ lớn.

---

### 3.2. Root Mean Squared Logarithmic Error (RMSLE)
RMSLE biến đổi cả giá trị thực tế và giá trị dự đoán thông qua hàm logarit tự nhiên trước khi tính bình phương sai số.

* **Công thức toán học:**
  $$\mathcal{L}_{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\ln(y_i + 1) - \ln(f(x_i) + 1)\right)^2}$$

* **Đặc tính cốt lõi:**
  * **Cộng $1$ ($\ln(x + 1)$):** Đảm bảo phép toán xác định khi dữ liệu chứa giá trị $0$.
  * **Nén thang đo (Scale Compression):** Phù hợp xuất sắc với dữ liệu trải dài qua nhiều bậc độ lớn hoặc có quy luật tăng trưởng hàm mũ (ví dụ: lượng người dùng truy cập, lưu lượng traffic, doanh thu lớn).
  * **Tập trung vào sai số tương đối:** Phạt nặng lỗi dự đoán thiếu (Underestimation) hơn lỗi dự đoán thừa (Overestimation).
  * **Hạn chế:** Không áp dụng được cho tập dữ liệu chứa giá trị âm ($y_i < 0$ hoặc $f(x_i) < 0$).

---

## 4. Nhóm Hàm Loss Dung hòa & Nâng cao ($L_1$ & $L_2$ Balance)

### 4.1. Huber Loss (Parametric)
Huber Loss là hàm mất mát từng khúc (piecewise), kết hợp ưu điểm của MSE ở vùng sai số nhỏ và MAE ở vùng sai số lớn.

* **Công thức toán học:**
  $$L_{Huber} = \begin{cases} \frac{1}{2}(y_i - f(x_i))^2 & \text{khi } |y_i - f(x_i)| \le \delta \\ \delta\left(|y_i - f(x_i)| - \frac{1}{2}\delta\right) & \text{ngược lại} \end{cases}$$

* **Cơ chế hoạt động:**
  * **Vùng sai số nhỏ ($|e_i| \le \delta$):** Hoạt động như MSE (bình phương sai số), giúp gradient giảm đều về $0$ và hội tụ nhanh, trơn tru.
  * **Vùng sai số lớn ($|e_i| > \delta$):** Hoạt động như MAE (tuyến tính), hạn chế tác động phá hoại của các điểm ngoại lai.
  * **Khả vi:** Trơn tru và có đạo hàm tại mọi điểm trên trục số.
  * **Hạn chế:** Cần phải thử nghiệm tinh chỉnh siêu tham số ngưỡng $\delta$.

> [!NOTE] Cần ảnh minh họa: Đồ thị hàm Huber Loss với các đường cong ứng với từng giá trị ngưỡng $\delta$ khác nhau ($\delta = 0.1, 0.2, 0.4, 0.6, 1.5$).

---

### 4.2. Smooth L1 Loss
Là một trường hợp cụ thể của Huber Loss với việc cố định cứng tham số $\delta = 1$.

* **Công thức toán học:**
  $$L_{Smooth\ L1} = \begin{cases} \frac{1}{2}(y_i - f(x_i))^2 & \text{khi } |y_i - f(x_i)| \le 1 \\ |y_i - f(x_i)| - \frac{1}{2} & \text{ngược lại} \end{cases}$$

* **Ứng dụng thực tế:** Được sử dụng rộng rãi trong các mạng học sâu phát hiện vật thể (như Fast R-CNN, Faster R-CNN) cho nhiệm vụ hồi quy tọa độ khung bao (Bounding Box Regression) nhằm đảm bảo tính ổn định gradient khi lan truyền ngược.

> [!NOTE] Cần ảnh minh họa: Đồ thị so sánh trực quan giữa 3 hàm: L1 (MAE), L2 (MSE) và Smooth L1.

---

### 4.3. Log-Cosh Loss (Non-parametric)
Log-Cosh tính logarit của hàm cosin hyperbol của sai số. Đây là giải pháp thay thế hoàn hảo cho Huber Loss khi không muốn phải chọn siêu tham số $\delta$.

* **Công thức toán học:**
  $$\mathcal{L}_{logcosh} = \frac{1}{N} \sum_{i=1}^{N} \ln(\cosh(f(x_i) - y_i))$$

* **Đặc tính cơ chế:**
  * Với sai số nhỏ: $\ln(\cosh(x)) \approx \frac{x^2}{2}$ (xấp xỉ MSE).
  * Với sai số lớn: $\ln(\cosh(x)) \approx |x| - \ln(2)$ (xấp xỉ MAE).
  * **Ưu điểm vượt trội:** Hoàn toàn trơn, có đạo hàm cấp hai liên tục ($\mathcal{C}^2$) và không cần cấu hình siêu tham số nào.

> [!NOTE] Cần ảnh minh họa: Đồ thị hình học của các hàm lượng giác hyperbol $\cosh(x)$, $\sinh(x)$, $\tanh(x)$ và hình dạng đường cong Log-Cosh so sánh với MSE, Huber.

---

## 5. Bảng So sánh Tổng hợp các Regression Loss

| Hàm Loss | Công thức cốt lõi | Khả vi tại 0? | Nhạy cảm Outlier | Đơn vị đo | Trường hợp nên dùng |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **MBE** | $\frac{1}{N}\sum (y - \hat{y})$ | Có | Trung bình | Gốc | Đo độ lệch thiên vị hệ thống, Calibration. |
| **MSE** | $\frac{1}{N}\sum (y - \hat{y})^2$ | Có | Rất cao | $(Đơn\ vị)^2$ | Dữ liệu sạch, cần phạt nặng các lỗi lớn. |
| **MAE** | $\frac{1}{N}\sum \|y - \hat{y}\|$ | Không | Thấp | Gốc | Dữ liệu có nhiều nhiễu, có ngoại lai. |
| **RMSE** | $\sqrt{\frac{1}{N}\sum (y - \hat{y})^2}$ | Có | Rất cao | Gốc | Cần đặc tính phạt của MSE nhưng đơn vị dễ hiểu. |
| **RMSLE** | $\sqrt{\frac{1}{N}\sum (\Delta \ln)^2}$ | Có | Rất thấp | Tương đối | Target trải rộng nhiều bậc độ lớn, không âm. |
| **Huber** | Piecewise ($L_2 \leftrightarrow L_1$) | Có | Thấp | Linh hoạt | Muốn dung hòa $L_1$ và $L_2$, chấp nhận tune $\delta$. |
| **Smooth L1** | Huber với $\delta = 1$ | Có | Thấp | Linh hoạt | Tác vụ Deep Learning / Object Detection. |
| **Log-Cosh** | $\frac{1}{N}\sum \ln(\cosh(\text{err}))$ | Có | Thấp | Linh hoạt | Cần sự dung hòa $L_1 - L_2$ mà không cần tune tham số. |

---

