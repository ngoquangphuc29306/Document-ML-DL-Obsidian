---
tags: [machine-learning, regression, polynomial-regression, artificial-intelligence]
aliases: [Hồi quy phi tuyến, Polynomial Regression]
date: 2026-08-29
---
---


# Non-linear Regression

## 1. Động lực và Hạn chế của Linear Regression
* **Hạn chế của Linear Regression:** Nhược điểm chính của Linear Regression là mô hình giả định tính tuyến tính ở cả tham số (parameters) và đặc trưng (features). Đây là một giả định rất hạn chế vì dữ liệu thực tế thường có các hành vi phi tuyến tính đối với các đặc trưng.
* **Giải pháp:** Cần mở rộng cách tiếp cận này để tạo ra các mô hình linh hoạt hơn, vượt ra khỏi các giới hạn tuyến tính thông thường.
![Pasted image 20260829145517](../../../05_Assets/Images/Pasted%20image%2020260829145517.png)

---
## 2. Bản chất của Non-linear Regression
* Non-linear Regression ước lượng đầu ra dựa trên các hàm phi tuyến tính, ví dụ như hàm đa thức (polynomial), hàm hình sin (sinusoidal), hàm bước (step function) hoặc hàm mũ (exponential).
* **Điểm mấu chốt:** Mô hình dự đoán vẫn là tuyến tính đối với các *tham số* (parameters) nhưng phi tuyến tính đối với các *đặc trưng* (features).
![Pasted image 20260829145534](../../../05_Assets/Images/Pasted%20image%2020260829145534.png)

---
## 3. Polynomial Regression (Hồi quy đa thức) và Biểu diễn dữ liệu
* Phương pháp này tạo ra các đặc trưng đa thức (Polynomial Features) thông qua một hàm cơ sở (basis function) $\psi(\cdot)$ dùng để biến đổi đầu vào, ví dụ như hàm lũy thừa.
* **Phương trình cho đa thức bậc b:** 
  $$\hat{y}(i) = \theta_0*1 + \theta_1*\varphi(i) + \theta_2*\varphi(i)^2 + ... + \theta_b*\varphi(i)^b$$
* Cả mô hình Linear Regression và Nonlinear Regression (cụ thể là Polynomial Regression) đều có tính tuyến tính trong các tham số.
* **Mở rộng cho nhiều biến (Multivariable):** Có thể mở rộng việc tạo đặc trưng cho nhiều biến, ví dụ áp dụng quy tắc khai triển đa thức: 
  $$(a+b)^2 \Rightarrow a^2 + b^2 + ab + a + b + 1$$

![Pasted image 20260829145624](../../../05_Assets/Images/Pasted%20image%2020260829145624.png)
![Pasted image 20260829145641](../../../05_Assets/Images/Pasted%20image%2020260829145641.png)


---
## 4. Lựa chọn bậc đa thức (Degree Choice) và Hiện tượng Overfitting
* Việc chọn bậc đa thức (degree) là vô cùng quan trọng và phụ thuộc vào từng tập dữ liệu.
* **Phân tích các trường hợp bậc đa thức:**
    * **Bậc quá thấp** (ví dụ: bậc 1 hoặc 2): Mô hình quá đơn giản, không đủ độ linh hoạt.
    * **Bậc phù hợp** (ví dụ: bậc 3): Vừa vặn với dữ liệu (Just right).
    * **Bậc quá cao** (ví dụ: bậc 9 hoặc 20): Gây ra hiện tượng quá khớp (Overfitting).
* **Phương pháp chọn bậc tối ưu:** Sử dụng K-fold cross-validation và chọn bậc mang lại sai số ngoài tập mẫu (out-of-sample error) thấp nhất.
* **Rủi ro:** Việc tăng bậc của đa thức luôn dẫn đến việc mô hình nhạy cảm hơn với nhiễu ngẫu nhiên (stochastic noise), đặc biệt là ở các vùng ranh giới nơi thường có ít dữ liệu hơn.

![Pasted image 20260829145734](../../../05_Assets/Images/Pasted%20image%2020260829145734.png)

---
## 5. Quá trình Huấn luyện (Modeling) bằng Gradient Descent
* **Thuật toán tối ưu:** Sử dụng thuật toán Gradient Descent để cập nhật tham số $\theta$ với công thức cập nhật:
  $$\theta = \theta - \eta\frac{L_{\theta}^{\prime}}{N}$$ 
  Trong đó $\eta$ là tốc độ học (learning rate).
* **Các bước thực hiện trong vòng lặp huấn luyện:** 
    1. Lấy mẫu từ tập dữ liệu huấn luyện.
    2. Tính toán đầu ra dự đoán $\hat{y}$.
    3. Tính toán hàm mất mát (Loss).
    4. Tính đạo hàm của hàm mất mát.
    5. Cập nhật các tham số mô hình.
* **Tốc độ học (Learning Rate):** Cần tìm tốc độ học tối ưu (Optimal). Nếu $\eta$ quá nhỏ, mô hình sẽ hội tụ chậm (Slow); nếu $\eta$ quá lớn, có thể khiến mô hình hội tụ kém (High).
![Pasted image 20260829150325](../../../05_Assets/Images/Pasted%20image%2020260829150325.png)
---
## 6. Quy trình Machine Learning thực tiễn (Ví dụ Fish Dataset)
Quy trình chuẩn khi áp dụng Non-linear Regression vào thực tế bao gồm các bước tuần tự:
1. **Preprocessing (Tiền xử lý):** Tiền xử lý dữ liệu thô, ví dụ sử dụng One Hot Encoding cho biến phân loại (Category).
2. **EDA (Exploratory Data Analysis):** Phân tích khám phá dữ liệu, chẳng hạn sử dụng Correlation Matrix.
3. **Representation:** Biểu diễn dữ liệu, bao gồm việc tạo ra các đặc trưng đa thức (Polynomial Features).
4. **Modelling:** Huấn luyện mô hình.
5. **Deployment:** Triển khai mô hình.

---
## 7. Triển khai bằng Python và NumPy (Từ con số 0 - Không dùng sklearn)
Để hiểu rõ hơn về bản chất của Gradient Descent và cách tạo đặc trưng đa thức được đề cập trong các phần trước, dưới đây là cách triển khai hoàn toàn bằng NumPy dựa trên các công thức toán học cốt lõi.

**Đoạn code minh họa:**
```python
import numpy as np

# 1. Hàm tạo đặc trưng đa thức (Polynomial Features)
def create_polynomial_features(X, degree=2):
    """
    Tạo các đặc trưng đa thức từ ma trận đầu vào X.
    X: mảng 2D kích thước (N, 1)
    """
    # Khởi tạo cột bias toàn số 1 (ứng với x^0)
    X_poly = np.ones((X.shape[0], 1)) 
    
    # Sinh các đặc trưng x^1, x^2, ..., x^degree
    for d in range(1, degree + 1):
        X_poly = np.c_[X_poly, np.power(X, d)]
        
    return X_poly

# === Dữ liệu giả lập (y xấp xỉ x^2) ===
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([[2.1], [3.9], [9.2], [16.1], [25.5]])

# 2. Tiền xử lý: Biến đổi dữ liệu thành đa thức bậc 2
degree = 2
X_poly = create_polynomial_features(X, degree)

# 3. Quá trình Huấn luyện (Gradient Descent)
learning_rate = 0.001
epochs = 10000
N = X_poly.shape[0] # Số lượng mẫu

# Khởi tạo tham số theta (trọng số) ngẫu nhiên
np.random.seed(42)
theta = np.random.randn(X_poly.shape[1], 1)

for epoch in range(epochs):
    # Bước 2: Tính toán đầu ra dự đoán (y_hat = X * theta)
    y_hat = X_poly.dot(theta)
    
    # Bước 4: Tính đạo hàm của hàm mất mát
    # L = (y_hat - y)^2 => L' = 2(y_hat - y)
    k = 2 * (y_hat - y)
    gradient = X_poly.T.dot(k) / N
    
    # Bước 5: Cập nhật tham số theta
    theta = theta - learning_rate * gradient

# === Kết quả ===
print("Các tham số theta tìm được (theta_0, theta_1, theta_2):")
print(theta)

# 4. Dự đoán trên tập dữ liệu đã huấn luyện
y_pred = X_poly.dot(theta)
print("\nGiá trị dự đoán:")
print(y_pred)
```

* **Ý nghĩa:** Đoạn code này mô phỏng chính xác vòng lặp học (Learning loop) đã mô tả ở **Phần 5**, bao gồm việc tính `y_hat`, tính sai số `k`, tính đạo hàm (Gradient) và cập nhật trọng số `theta`. Cách tiếp cận này cung cấp cái nhìn chi tiết nhất về cách mô hình thực sự hội tụ.

---
## 8. Triển khai bằng Python với thư viện scikit-learn (sklearn)
Trong thực tế, mô hình Polynomial Regression thường được triển khai dễ dàng thông qua thư viện `scikit-learn`. Bản chất của phương pháp này là sử dụng `PolynomialFeatures` để biến đổi dữ liệu đầu vào thành các đặc trưng đa thức, sau đó áp dụng mô hình `LinearRegression` tiêu chuẩn lên tập đặc trưng mới đó.

**Đoạn code minh họa:**
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Giả sử X, y là tập dữ liệu đầu vào và mục tiêu của bạn
# X = ... 
# y = ...

# 1. Khởi tạo bộ biến đổi đặc trưng đa thức (ví dụ: bậc 2)
poly_features = PolynomialFeatures(degree=2)

# 2. Biến đổi dữ liệu X ban đầu thành tập đặc trưng đa thức (X_poly)
# Sử dụng X.to_frame() hoặc mảng 2D nếu X là pandas Series/1D numpy array
X_poly = poly_features.fit_transform(X)

# Cấu trúc của X_poly lúc này sẽ bao gồm: 1, x, x^2 (với include_bias mặc định là True)

# 3. Khởi tạo và huấn luyện mô hình Linear Regression trên tập đặc trưng mới
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# 4. Dự đoán kết quả
y_pred = lin_reg.predict(X_poly)
```

* **Lưu ý:** Hàm `PolynomialFeatures` giúp tự động sinh ra các tổ hợp đặc trưng bậc cao và đặc trưng tương tác. Ví dụ, nếu đầu vào có 2 biến $a, b$ và `degree=2`, hàm sẽ tạo ra tập đặc trưng: $[1, a, b, a^2, ab, b^2]$.