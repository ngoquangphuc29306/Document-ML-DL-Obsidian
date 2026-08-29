---
tags: [ml/preprocessing, feature-engineering, data-transformation]
created: 2026-07-21
---

# Feature Scaling (Chuẩn hóa đặc trưng)

**Feature Scaling** là một kỹ thuật trong bước Tiền xử lý dữ liệu (Data Preprocessing), nhằm đưa phạm vi giá trị (scale/range) của các đặc trưng (features) về cùng một quy chuẩn chung mà **không làm thay đổi hình dạng phân bố hoặc mối quan hệ** giữa các điểm dữ liệu.

---

## 1. Tại sao cần Feature Scaling?

Khi các đặc trưng có đơn vị hoặc quy mô khác biệt lớn (ví dụ: `Tuổi` từ 18-65, nhưng `Thu nhập` từ 5,000,000 - 100,000,000):

1. **Tránh sự áp đảo của các đặc trưng có biên độ lớn:** 
   Các thuật toán tính toán dựa trên thước đo khoảng cách (như [[Similarity Metrics#1 1 Euclidean Distance Khoảng cách L_2|Euclidean Distance]]) trong [[K-Nearest Neighbors]] hay [[K-Means Clustering]] sẽ bị lệch hoàn toàn (Bias) theo đặc trưng có biên độ lớn hơn.
2. **Tăng tốc độ hội tụ (Convergence):**
   Với các thuật toán tối ưu bằng Gradient Descent (như Logistic Regression, Neural Networks), việc Feature Scaling giúp mặt hàm mất mát (Loss Landscape) tròn hơn, giúp thuật toán hội tụ về điểm tối ưu nhanh hơn nhiều.

---

## 2. Hai phương pháp Feature Scaling phổ biến nhất

### 2.1. Standardization (Z-Score Normalization)

Phương pháp này biến đổi dữ liệu sao cho phân bố có **giá trị trung bình $\mu = 0$** và **độ lệch chuẩn $\sigma = 1$**.

#### Công thức toán học:
$$x' = \frac{x - \mu}{\sigma}$$

* $\mu$ (Mean): Giá trị trung bình của đặc trưng.
* $\sigma$ (Standard Deviation): Độ lệch chuẩn của đặc trưng.

#### Đặc điểm:
* Dữ liệu sau khi scaling không bị giới hạn trong một khoảng cố định (có thể âm hoặc dương).
* **Kháng Outliers tốt hơn Min-Max:** Do không phụ thuộc hoàn toàn vào giá trị Max/Min tuyệt đối.
* Phù hợp khi dữ liệu có dạng **Phân bố chuẩn (Gaussian/Normal Distribution)**.

---

### 2.2. Min-Max Scaling (Normalization)

Phương pháp này biến đổi toàn bộ dữ liệu về một khoảng cố định, thường là **$[0, 1]$** (hoặc $[-1, 1]$).

#### Công thức toán học:
$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

* $x_{\min}$: Giá trị nhỏ nhất của đặc trưng.
* $x_{\max}$: Giá trị lớn nhất của đặc trưng.

#### Đặc điểm:
* Đảm bảo chính xác mọi giá trị nằm trong khoảng $[0, 1]$.
* **Rất nhạy cảm với Outliers:** Nếu dữ liệu có $x_{\max}$ cực lớn (nhiễu), toàn bộ dữ liệu bình thường sẽ bị nén chặt về một khoảng cực nhỏ sát $0$.
* Phù hợp khi bạn biết chắc giới hạn trên/dưới của dữ liệu hoặc khi thuật toán yêu cầu đầu vào trong $[0, 1]$ (ví dụ: Hình ảnh - Pixel intensity $0 \rightarrow 255$).

---

## 3. Bảng so sánh các phương pháp Scaling

| Tiêu chí | Standardization (Z-Score) | Min-Max Scaling |
| :--- | :--- | :--- |
| **Phạm vi đầu ra** | Không giới hạn (Thường trong $[-3, 3]$) | Cố định $[0, 1]$ (hoặc $[-1, 1]$) |
| **Độ nhạy với Outliers** | Ít bị ảnh hưởng hơn | Rất nhạy cảm |
| **Giữ nguyên dạng phân bố gốc** | Có | Có |
| **Khi nào nên dùng?** | - Dữ liệu dạng Phân bố chuẩn.<br>- Thuật toán: KNN, K-Means, PCA, SVM, Neural Networks. | - Dữ liệu không tuân theo phân bố chuẩn.<br>- Xử lý ảnh (Pixel), Thuật toán yêu cầu dải $[0, 1]$. |

---

## 4. Thuật toán nào CẦN và KHÔNG CẦN Feature Scaling?

* **CẦN Feature Scaling:**
  * Thuật toán dựa trên khoảng cách: [[K-Nearest Neighbors]], [[K-Means Clustering]], [[Support Vector Machines (SVM)]].
  * Thuật toán giảm chiều dữ liệu dựa trên phương sai: [[PCA]] (Principal Component Analysis).
  * Thuật toán dùng Gradient Descent: [[Logistic Regression]], [[Linear Regression]], [[Neural Networks]].

* **KHÔNG CẦN Feature Scaling:**
  * Các thuật toán dựa trên cây (Tree-based Models): [[Decision Tree]], [[Random Forest]], [[XGBoost]], [[Gradient Boosting]]. *(Lý do: Cây quyết định chia nhánh dựa trên các ngưỡng $X_j \ge \text{threshold}$ độc lập từng đặc trưng, không phụ thuộc vào tỉ lệ scale giữa các đặc trưng với nhau).*

---

## 5. CẢNH BÁO: Rò rỉ dữ liệu (Data Leakage) khi Scaling

> ⚠️ **Quy tắc vàng:** Chỉ tính toán $\mu, \sigma$ (hoặc $x_{\min}, x_{\max}$) trên **tập Train**, sau đó dùng thông số đó để `transform` cho tập Validation/Test. **KHÔNG BAO GIỜ** scale toàn bộ dataset trước khi chia tập Train/Test!

### Cách viết đúng với Python Scikit-Learn:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. Chia tập Train/Test trước
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Khởi tạo Scaler
scaler = StandardScaler()

# 3. FIT + TRANSFORM trên tập TRAIN
X_train_scaled = scaler.fit_transform(X_train)

# 4. CHỈ TRANSFORM trên tập TEST (dùng tham số đã fit ở train)
X_test_scaled = scaler.transform(X_test)