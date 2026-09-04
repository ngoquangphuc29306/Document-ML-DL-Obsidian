---
tags:
  - machine-learning
  - linear-regression
  - supervised-learning
  - optimization
  - python
  - numpy
---

# Linear Regression: Từ Cơ Bản Đến Nâng Cao

> [!abstract] Tổng quan
> **Linear Regression (Hồi quy tuyến tính)** là một thuật toán học máy giám sát (Supervised Learning) thuộc bài toán hồi quy. Mục tiêu của thuật toán là tìm ra một đường thẳng (hoặc siêu mặt phẳng) khớp nhất với dữ liệu để dự đoán một giá trị liên tục ($\hat{y}$) dựa trên các đặc trưng đầu vào ($X$).

---

## 1. Mô Hình Toán Học

### 1.1. Hồi Quy Tuyến Tính Đơn Biến (Simple Linear Regression)
Dự đoán đầu ra dựa trên một đặc trưng duy nhất.
$$\hat{y} = wx + b$$
*   **$w$ (Weight/Slope):** Trọng số, thể hiện độ dốc của đường thẳng.
*   **$b$ (Bias/Intercept):** Điểm cắt trục tung, độ lệch của đường thẳng.
*   **$x$:** Đặc trưng đầu vào.

![Pasted image 20260829144707](../../../05_Assets/Images/Pasted%20image%2020260829144707.png)

### 1.2. Hồi Quy Tuyến Tính Đa Biến & Vector Hóa (Multivariate & Vectorization)
Dự đoán đầu ra dựa trên nhiều đặc trưng, được biểu diễn dưới dạng đại số tuyến tính để tối ưu hóa hiệu suất tính toán.

> [!tip] Lý do cần Vector hóa
> Thay thế các vòng lặp `for` truyền thống bằng phép toán ma trận giúp tận dụng sức mạnh tính toán song song, tăng tốc độ huấn luyện mô hình lên gấp nhiều lần.

Mô hình cho $N$ mẫu dữ liệu:
$$\hat{Y} = X\theta$$
Trong đó:
*   **$\hat{Y}$**: Vector dự đoán kích thước $(N \times 1)$.
*   **$X$**: Ma trận đầu vào kích thước $(N \times (d+1))$, với $d$ là số đặc trưng. Cột đầu tiên thường được thêm toàn giá trị $1$ để nhân với Bias.
*   **$\theta$**: Vector tham số kích thước $((d+1) \times 1)$, bao gồm cả Bias $b$ ở phần tử đầu tiên và các trọng số $w_1, w_2, ...$

---

## 2. Hàm Mất Mát (Loss Functions)

Hàm mất mát đo lường mức độ sai lệch giữa dự đoán ($\hat{y}$) và thực tế ($y$). Dưới đây là 3 hàm mất mát cốt lõi:

### 2.1. Mean Squared Error (MSE / Squared Loss)
Đo lường trung bình bình phương sai số.
*   **Công thức (1 mẫu):** $L(\hat{y}, y) = \frac{1}{2} (\hat{y} - y)^2$
*   **Công thức (N mẫu):** $L = \frac{1}{N} (\hat{y} - Y)^T (\hat{y} - Y)$
*   **Đạo hàm theo $w$:** $\frac{\partial L}{\partial w} = 2x(\hat{y} - y)$

![Pasted image 20260829144825](../../../05_Assets/Images/Pasted%20image%2020260829144825.png)
> [!success] Ưu điểm
> Đạo hàm mượt mà, hội tụ tốt và dễ tối ưu bằng Gradient Descent. Phạt rất nặng các điểm sai số lớn.
> [!bug] Nhược điểm
> Cực kỳ nhạy cảm với dữ liệu nhiễu (outliers). Một outlier có thể kéo lệch toàn bộ đường hồi quy.

### 2.2. Mean Absolute Error (MAE)
Đo lường trung bình giá trị tuyệt đối của sai số.
*   **Công thức (1 mẫu):** $L(\hat{y}, y) = \vert{}\hat{y} - y\vert{}$
![Pasted image 20260829144849](../../../05_Assets/Images/Pasted%20image%2020260829144849.png)

> [!success] Ưu điểm
> Kháng nhiễu (outliers) cực tốt vì sai số được tính theo tuyến tính, không bị phóng đại bằng phép bình phương.

> [!bug] Nhược điểm
> Không khả vi (không có đạo hàm) tại điểm sai số bằng $0$. Quá trình hội tụ ở gần điểm cực tiểu khó khăn hơn.

### 2.3. Huber Loss
Kết hợp hoàn hảo giữa điểm mạnh của MSE và MAE.
*   **Công thức:**
$$L_\delta(\hat{y}, y) = \begin{cases} \frac{1}{2}(\hat{y} - y)^2 & \text{nếu } \vert{}\hat{y} - y\vert{} \le \delta \\ \delta \vert{}\hat{y} - y\vert{} - \frac{1}{2}\delta^2 & \text{nếu } \vert{}\hat{y} - y\vert{} > \delta \end{cases}$$
*   **Đạo hàm theo $w$:**
$$\frac{\partial L}{\partial w} = \begin{cases} x(\hat{y} - y) & \text{nếu } \vert{}\hat{y} - y\vert{} \le \delta \\ \delta x \cdot \frac{\hat{y} - y}{\vert{}\hat{y} - y\vert{}} & \text{nếu } \vert{}\hat{y} - y\vert{} > \delta \end{cases}$$

![Pasted image 20260829145003](../../../05_Assets/Images/Pasted%20image%2020260829145003.png)
> [!info] Đặc điểm
> Sử dụng hành vi của MSE khi sai số nhỏ (dưới ngưỡng $\delta$) để dễ hội tụ, và sử dụng hành vi của MAE khi sai số lớn (vượt ngưỡng $\delta$) để kháng outliers.

---

## 3. Thuật Toán Gradient Descent & Learning Rate

### 3.1. Nguyên lý hoạt động
Gradient Descent tối ưu hóa hàm mất mát bằng cách di chuyển các tham số ngược hướng với vector đạo hàm tại mỗi vòng lặp (epoch).
$$\theta_{new} = \theta_{old} - \eta \nabla_\theta L$$

*Với tập N mẫu (Vector hóa):*
*   **Đạo hàm (Gradient):** $\nabla_\theta L = X^T k$ *(với $k = \frac{2}{N}(\hat{y} - y)$)*
*   **Cập nhật:** $\theta = \theta - \eta \nabla_\theta L$
![Pasted image 20260829145201](../../../05_Assets/Images/Pasted%20image%2020260829145201.png)

### 3.2. Vai trò của Learning Rate ($\eta$)
$\eta$ quyết định độ dài bước nhảy trong không gian tham số.
*   **$\eta$ quá nhỏ:** Mô hình cần rất nhiều bước (epochs) để hội tụ $\rightarrow$ Tốn thời gian huấn luyện.
*   **$\eta$ quá lớn:** Thuật toán bước những bước quá dài, vượt qua (overshoot) điểm cực tiểu, gây ra dao động mạnh, hoặc thậm chí phân kỳ (Loss tiến tới vô cực).

![Pasted image 20260829145122](../../../05_Assets/Images/Pasted%20image%2020260829145122.png)
### 3.3. So sánh MSE và MAE với Learning Rate cố định
> [!example] Tại sao MSE hoạt động tốt hơn MAE khi $\eta$ cố định?
> *   **Với MAE:** Đạo hàm luôn có độ lớn hằng số (vd: $1$ hoặc $-1$). Bước nhảy $\eta \nabla L$ sẽ có độ lớn không đổi. Khi tiến sát cực tiểu, mô hình dễ bị nhảy qua lại (dao động) qua điểm tối ưu mà không thể chạm đáy nếu không giảm $\eta$.
> *   **Với MSE:** Đạo hàm tỉ lệ thuận với sai số ($\nabla L \propto (\hat{y} - y)$). Khi ở xa đáy, sai số lớn $\rightarrow$ bước nhảy lớn (học nhanh). Khi tiến gần đáy, sai số nhỏ lại $\rightarrow$ đạo hàm và bước nhảy tự động thu nhỏ, giúp mô hình "hạ cánh" mượt mà vào điểm cực tiểu.

---

## 4. Các Chiến Lược Huấn Luyện (Training Strategies)

| Chiến lược | Đặc điểm | Ưu điểm | Nhược điểm |
| :--- | :--- | :--- | :--- |
| **Stochastic GD (SGD)** | Cập nhật tham số sau *mỗi 1 mẫu* dữ liệu. | Tính toán rất nhanh cho mỗi bước nhảy. | Quá trình giảm Loss bị nhiễu, dao động cực mạnh. |
| **Mini-batch GD** | Cập nhật sau mỗi *nhóm nhỏ ($m$ mẫu)*. | Cân bằng tốt giữa tốc độ và độ mượt của quỹ đạo hội tụ. Tận dụng được phép toán ma trận. | Cần điều chỉnh thêm siêu tham số $m$ (batch size). |
| **Batch GD** | Tính Gradient và cập nhật trên *toàn bộ tập dữ liệu ($N$ mẫu)*. | Đảm bảo quỹ đạo hội tụ ổn định nhất, hướng thẳng về cực tiểu. | Tính toán chậm và ngốn rất nhiều RAM nếu tập dữ liệu khổng lồ. |

---

## 5. Tầm Quan Trọng Của Tiền Xử Lý Dữ Liệu (Data Normalization)
> [!warning] Chú ý
> Nếu các đặc trưng có thang đo (scale) chênh lệch quá lớn (VD: $x_1$ đếm số phòng từ 1-5, $x_2$ đo diện tích từ 50-200m²), bề mặt không gian của hàm mất mát sẽ bị kéo giãn thành hình elip hẹp.
> $\rightarrow$ Gradient Descent sẽ dao động mạnh qua lại theo trục ngắn và hội tụ cực kỳ chậm. Việc **Chuẩn hóa (Normalization/Standardization)** đưa dữ liệu về cùng thang đo giúp bề mặt Loss tròn trịa hơn, mô hình hội tụ nhanh chóng và ổn định.

---

## 6. Quy Trình Tổng Quát (Pipeline) Xây Dựng Linear Regression (Vector Hóa)

> [!info] Quy ước ký hiệu (Notation Convention)
> *   **$d$**: Số lượng đặc trưng (features).
> *   **Vô hướng (Scalar):** Chữ cái in nghiêng viết thường (VD: $y, \hat{y}, L, \eta, m, N$).
> *   **Vector & Ma trận:** Chữ in đậm (VD: $\mathbf{x}, \mathbf{X}, \mathbf{y}, \mathbf{\hat{y}}, \boldsymbol{\theta}, \mathbf{k}$).
> *   **Vector tham số $\boldsymbol{\theta}$**: Khởi tạo ban đầu có kích thước $(d+1) \times 1$:
>     $$ \boldsymbol{\theta} = \begin{bmatrix} b \\ w_1 \\ \vdots \\ w_d \end{bmatrix} $$

---

### 6.1. Huấn luyện với 1 mẫu (1-Sample / Stochastic Gradient Descent)

Cập nhật tham số ngay sau khi tính toán trên từng mẫu đơn lẻ.

1.  **Lấy mẫu (Pick a sample):** Rút ra 1 vector đặc trưng $\mathbf{x}$ (thêm $1$ cho Bias) và nhãn $y$ (scalar):
    $$ \mathbf{x} = \begin{bmatrix} 1 \\ x_1 \\ \vdots \\ x_d \end{bmatrix}, \quad y = \text{giá trị thực tế} $$
2.  **Dự đoán (Forward):** Tính giá trị dự đoán $\hat{y}$ (scalar) bằng tích vô hướng:
    $$ \hat{y} = \mathbf{x}^T \boldsymbol{\theta} = \begin{bmatrix} 1 & x_1 & \dots & x_d \end{bmatrix} \begin{bmatrix} b \\ w_1 \\ \vdots \\ w_d \end{bmatrix} = b + w_1x_1 + \dots + w_dx_d $$
3.  **Tính Loss (Compute Loss):** Mean Squared Error cho 1 mẫu:
    $$ L = (\hat{y} - y)^2 $$
4.  **Tính Đạo hàm (Compute Derivative):** Vector Gradient $\nabla_{\boldsymbol{\theta}} L$ nhân vô hướng sai số với vector $\mathbf{x}$[cite: 5]:
    $$ \nabla_{\boldsymbol{\theta}} L = 2\mathbf{x}(\hat{y} - y) = \begin{bmatrix} 2(\hat{y}-y) \cdot 1 \\ 2(\hat{y}-y) \cdot x_1 \\ \vdots \\ 2(\hat{y}-y) \cdot x_d \end{bmatrix} = \begin{bmatrix} \frac{\partial L}{\partial b} \\ \frac{\partial L}{\partial w_1} \\ \vdots \\ \frac{\partial L}{\partial w_d} \end{bmatrix} $$
5.  **Cập nhật tham số (Update Parameters):** Điều chỉnh vector $\boldsymbol{\theta}$[:
    $$ \boldsymbol{\theta} = \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} L $$

![Pasted image 20260829145328](../../../05_Assets/Images/Pasted%20image%2020260829145328.png)

---

### 6.2. Huấn luyện với nhóm $m$ mẫu (m-Samples / Mini-batch Gradient Descent)

Lấy một nhóm nhỏ $m$ mẫu ($1 < m < N$) để tận dụng tính toán ma trận, giúp hội tụ mượt mà và nhanh hơn.

1.  **Lấy mẫu (Pick m samples):** Lấy $m$ mẫu tạo thành ma trận $\mathbf{X}$ và vector nhãn $\mathbf{y}$:
    $$ \mathbf{X} = \begin{bmatrix} 1 & x_1^{(1)} & \dots & x_d^{(1)} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_1^{(m)} & \dots & x_d^{(m)} \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} y^{(1)} \\ \vdots \\ y^{(m)} \end{bmatrix} $$
2.  **Dự đoán (Forward):** Tính vector dự đoán $\mathbf{\hat{y}}$ kích thước $m \times 1$:
    $$ \mathbf{\hat{y}} = \mathbf{X}\boldsymbol{\theta} = \begin{bmatrix} b + w_1 x_1^{(1)} + \dots + w_d x_d^{(1)} \\ \vdots \\ b + w_1 x_1^{(m)} + \dots + w_d x_d^{(m)} \end{bmatrix} = \begin{bmatrix} \hat{y}^{(1)} \\ \vdots \\ \hat{y}^{(m)} \end{bmatrix} $$
3.  **Tính Loss (Compute Loss):** MSE trung bình cho $m$ mẫu:
    $$ L = \frac{1}{m} (\mathbf{\hat{y}} - \mathbf{y})^T (\mathbf{\hat{y}} - \mathbf{y}) $$
4.  **Tính Đạo hàm (Compute Derivative):** 
    * Tính vector chênh lệch $\mathbf{k}$ ($m \times 1$):
      $$ \mathbf{k} = 2(\mathbf{\hat{y}} - \mathbf{y}) = \begin{bmatrix} 2(\hat{y}^{(1)} - y^{(1)}) \\ \vdots \\ 2(\hat{y}^{(m)} - y^{(m)}) \end{bmatrix} $$
    * Tính vector Gradient $\nabla_{\boldsymbol{\theta}} L$ bằng phép nhân ma trận:
      $$ \nabla_{\boldsymbol{\theta}} L = \mathbf{X}^T \mathbf{k} = \begin{bmatrix} 1 & \dots & 1 \\ x_1^{(1)} & \dots & x_1^{(m)} \\ \vdots & \ddots & \vdots \\ x_d^{(1)} & \dots & x_d^{(m)} \end{bmatrix} \begin{bmatrix} k^{(1)} \\ \vdots \\ k^{(m)} \end{bmatrix} = \begin{bmatrix} \sum_{i=1}^m k^{(i)} \\ \sum_{i=1}^m k^{(i)}x_1^{(i)} \\ \vdots \\ \sum_{i=1}^m k^{(i)}x_d^{(i)} \end{bmatrix} $$
5.  **Cập nhật tham số (Update Parameters):** Lấy trung bình Gradient của $m$ mẫu:
    $$ \boldsymbol{\theta} = \boldsymbol{\theta} - \eta \frac{\nabla_{\boldsymbol{\theta}} L}{m} $$

![Pasted image 20260829145352](../../../05_Assets/Images/Pasted%20image%2020260829145352.png)

---

### 6.3. Huấn luyện với toàn bộ $N$ mẫu (N-Samples / Batch Gradient Descent)

Duyệt qua toàn bộ tập dữ liệu gồm $N$ mẫu trong một lần lặp. Về mặt ma trận, công thức hoàn toàn tương tự Mini-batch nhưng kích thước hàng mở rộng ra $N$.

1.  **Lấy dữ liệu (Pick all N samples):** Sử dụng toàn bộ ma trận đặc trưng $\mathbf{X} \in \mathbb{R}^{N \times (d+1)}$ và vector nhãn $\mathbf{y} \in \mathbb{R}^{N \times 1}$:
    $$ \mathbf{X} = \begin{bmatrix} 1 & x_1^{(1)} & \dots & x_d^{(1)} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_1^{(N)} & \dots & x_d^{(N)} \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} y^{(1)} \\ \vdots \\ y^{(N)} \end{bmatrix} $$
2.  **Dự đoán (Forward):** Tính vector dự đoán $\mathbf{\hat{y}}$ cho toàn bộ $N$ mẫu:
    $$ \mathbf{\hat{y}} = \mathbf{X}\boldsymbol{\theta} = \begin{bmatrix} \hat{y}^{(1)} \\ \vdots \\ \hat{y}^{(N)} \end{bmatrix} $$
3.  **Tính Loss (Compute Loss):** Trung bình sai số toàn tập dữ liệu:
    $$ L = \frac{1}{N} (\mathbf{\hat{y}} - \mathbf{y})^T (\mathbf{\hat{y}} - \mathbf{y}) $$
4.  **Tính Đạo hàm (Compute Derivative):**
    * Vector chênh lệch $\mathbf{k}$:
      $$ \mathbf{k} = 2(\mathbf{\hat{y}} - \mathbf{y}) = \begin{bmatrix} 2(\hat{y}^{(1)} - y^{(1)}) \\ \vdots \\ 2(\hat{y}^{(N)} - y^{(N)}) \end{bmatrix} $$
    * Tích ma trận tính Gradient cho toàn bộ trọng số cùng một lúc:
      $$ \nabla_{\boldsymbol{\theta}} L = \mathbf{X}^T \mathbf{k} = \begin{bmatrix} \sum_{i=1}^N k^{(i)} \\ \sum_{i=1}^N k^{(i)}x_1^{(i)} \\ \vdots \\ \sum_{i=1}^N k^{(i)}x_d^{(i)} \end{bmatrix} $$
5.  **Cập nhật tham số (Update Parameters):** Lấy trung bình Gradient của $N$ mẫu:
    $$ \boldsymbol{\theta} = \boldsymbol{\theta} - \eta \frac{\nabla_{\boldsymbol{\theta}} L}{N} $$

![Pasted image 20260829145414](../../../05_Assets/Images/Pasted%20image%2020260829145414.png)

> [!tip] Ý nghĩa của phép nhân $\mathbf{X}^T \mathbf{k}$
> Phép nhân ma trận $\mathbf{X}^T \mathbf{k}$ bản chất là cách viết gộp (vector hóa) của việc lấy **tổng của (sai số $\times$ giá trị đặc trưng)** trên toàn bộ $m$ hoặc $N$ mẫu cho từng trọng số $w$ tương ứng. Nó giúp CPU/GPU tính toán toàn bộ vòng lặp nội tại cùng một lúc bằng một phép `.dot()`[cite: 5, 6].

## 7. Mẫu Cài Đặt Khung Mã Nguồn (Python / NumPy)

```python
import numpy as np

def train_linear_regression(X, y, epochs=1000, lr=0.01):
    N, d = X.shape
    # 1. Thêm cột Bias (giá trị 1) vào ma trận X
    X_bias = np.hstack([np.ones((N, 1)), X])
    
    # 2. Khởi tạo tham số theta
    theta = np.zeros((d + 1, 1))
    losses = []
    
    for epoch in range(epochs):
        # 3. Lan truyền xuôi: Dự đoán output
        y_hat = X_bias.dot(theta)
        
        # 4. Tính MSE Loss
        loss = np.mean((y_hat - y) ** 2)
        losses.append(loss)
        
        # 5. Tính Gradient theo kỹ thuật Vector hóa
        k = 2 * (y_hat - y)
        gradients = (X_bias.T.dot(k)) / N
        
        # 6. Cập nhật tham số
        theta = theta - lr * gradients
        
    return theta, losses
```

## 8. Triển Khai Bằng Thư Viện Scikit-Learn (sklearn)

> [!info] Ứng dụng thực tế
> Trong quá trình học thuật, việc tự xây dựng thuật toán bằng `NumPy` giúp bạn nắm vững bản chất toán học của quá trình đạo hàm và cập nhật ma trận. Tuy nhiên, khi làm việc trong các dự án AI thực tế, chúng ta hầu như luôn sử dụng thư viện **Scikit-Learn (`sklearn`)**. 
> Thư viện này cung cấp các thuật toán đã được tối ưu hóa cực hạn (viết trên nền C/Cython), giúp mã nguồn ngắn gọn, chuẩn mực và tốc độ thực thi vượt trội.

Với `sklearn`, bạn không cần phải tự viết vòng lặp cập nhật `epoch` hay tính đạo hàm thủ công. Thư viện cung cấp sẵn giao diện chuẩn hóa qua các phương thức `.fit()` và `.predict()`.

### Mẫu Code Cài Đặt (Python / Scikit-Learn)

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Chuẩn bị dữ liệu (Ví dụ: N = 4 mẫu, d = 2 đặc trưng)
# X: Ma trận đầu vào (N, d). CHÚ Ý: KHÔNG CẦN thêm cột số 1 cho Bias, sklearn sẽ tự làm.
X = np.array([
    [6.7, 12], 
    [4.6, 13], 
    [3.5, 14], 
    [5.5, 15]
])

# y: Vector nhãn thực tế
y = np.array([9.1, 5.9, 4.6, 6.7])

# 2. Khởi tạo mô hình Hồi quy tuyến tính
model = LinearRegression()

# 3. Huấn luyện mô hình (Tìm trọng số w và bias b tối ưu)
model.fit(X, y)

# 4. Lấy và in các tham số đã học được
print("Trọng số (Weights) - w1, w2:", model.coef_)
print("Độ lệch (Bias) - b:", model.intercept_)

# 5. Lan truyền xuôi: Dự đoán trên dữ liệu mới (hoặc chính dữ liệu huấn luyện)
y_pred = model.predict(X)
print("Giá trị dự đoán (y_hat):", y_pred)

# 6. Đánh giá mô hình bằng hàm mất mát MSE
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE):", mse)
```


> [!tip]
> SGD Regressor cho Dữ Liệu Khổng Lồ  
> Hàm `LinearRegression` hoạt động hoàn hảo cho bộ dữ liệu vừa và nhỏ (chạy Batch toàn tập).  
> Tuy nhiên, nếu ma trận $\mathbf{X}$ của bạn quá lớn (hàng chục triệu dòng, hàng nghìn đặc trưng) khiến RAM máy tính bị tràn khi tải toàn bộ ma trận, bạn nên chuyển sang dùng **`SGDRegressor`**.

```python
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    learning_rate='constant',
    eta0=0.01
)
```
