---
tags:
  - ensemble-learning
  - supervised-learning
  - regression
  - classification
date: 2026-07-23
---
# Random Forest

## 1. Định nghĩa Random Forest
- **Random Forest** là một thuật toán [[Ensemble Learning]] thuộc nhóm **Bagging**, kết hợp nhiều Decision Trees để tạo ra dự đoán ổn định hơn một cây đơn lẻ.
- Random Forest tạo sự đa dạng giữa các cây bằng hai cơ chế ngẫu nhiên chính:
	1. **Bootstrap Sampling:**  
	   Mỗi cây được huấn luyện trên một tập bootstrap lấy ngẫu nhiên có hoàn lại từ tập dữ liệu gốc.	
	2. **Random Feature Subsampling:**  
	   Tại mỗi node, thuật toán chỉ chọn ngẫu nhiên một tập con các feature, sau đó tìm phép chia tốt nhất trong tập feature đó.
Giả sử dữ liệu có tổng cộng $p$ feature. Tại mỗi node, chỉ có $m$ feature được xem xét:
$$
m \le p.
$$
Hai cơ chế này giúp các cây ít tương quan hơn, từ đó làm giảm variance của mô hình tổng hợp.
![Pasted image 20260723210745](../../../../05_Assets/Images/Pasted%20image%2020260723210745.png)

---
## 2. Cách Xây Dựng Random Forest (Classification & Regression)

### Thuật toán tổng quát:
1. Cho tập dữ liệu gốc có $N$ mẫu và $p$ đặc trưng.
2. Với mỗi cây $t = 1, 2, ..., B$ trong rừng:
   * **Bước 1:** Rút trích một tập dữ liệu Bootstrap $D_t$ kích thước $N$ từ tập gốc (lấy mẫu có hoàn lại).
   * **Bước 2:** Xây dựng một Cây quyết định trên $D_t$. Tại mỗi nút phân nhánh:
     * Chọn ngẫu nhiên $m$ đặc trưng từ $n$ đặc trưng ($m < n$).
     * Tìm đặc trưng và ngưỡng chia (split point) tốt nhất trong $m$ đặc trưng đó.
     * Tách nút và tiếp tục phát triển cây cho đến khi đạt điều kiện dừng (không cắt tỉa - untrimmed tree).

### Bootstrap Sample và Out-of-Bag Samples

- Mỗi bootstrap sample được tạo bằng cách lấy $N$ lần từ tập dữ liệu gồm $N$ mẫu, có hoàn lại.
	- Do có hoàn lại:
	- một mẫu có thể xuất hiện nhiều lần;
	- một số mẫu có thể không xuất hiện lần nào.
- Xác suất một mẫu không được chọn trong một lần rút là:
$$
1-\frac{1}{N}.
$$
- Sau $N$ lần rút, xác suất mẫu đó không được chọn là:
$$
\left(1-\frac{1}{N}\right)^N.
$$
- Khi $N$ lớn:
$$
\left(1-\frac{1}{N}\right)^N
\approx e^{-1}
\approx 0.368.
$$
- Do đó:
	- khoảng $36.8\%$ số mẫu không xuất hiện trong bootstrap sample;
	- khoảng $63.2\%$ số mẫu duy nhất xuất hiện ít nhất một lần.
- Các mẫu không được chọn cho cây $b$ được gọi là **Out-of-Bag samples** của cây đó.

### Tổng hợp kết quả đưa ra dự đoán (Aggregation):

#### A. Bài toán Phân loại (Classification)
* **Số đặc trưng được chọn ngẫu nhiên tại mỗi nút:**  $m = \lfloor \sqrt{p} \rfloor$ (giá trị đề xuất)
* **Thước đo điểm phân nhánh:** Sử dụng **Gini Impurity** hoặc **Entropy**.
* **Dự đoán cuối cùng:** Dùng cơ chế bầu chọn theo đa số (**Majority Voting**):
  $$\hat{y}_{RF}(\mathbf{p}_{new}) = \arg\max_{c} \sum_{b=1}^{B} I(\hat{y}_b(\mathbf{p}_{new}) = c)$$
  - Trong đó $I(\cdot)$ là hàm chỉ thị Indicator Function, trả về 1 nếu cây thứ $b$ dự đoán ra nhãn $c$ và 0 nếu ngược lại.
  - $\mathbf{p}_{new}$ là điểm dữ liệu mới.
##### Code Python
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# 1. Load data
iris = load_iris(as_frame=True)
df = iris.frame

# 2. Get data
x_data = df[["petal length (cm)", "petal width (cm)"]].to_numpy()
y_data = df["target"].to_numpy(dtype=np.uint8)

# 3. Define Model
rf_classifier = RandomForestClassifier(
    n_estimators=3,     # Số lượng decision tree
    max_features=1,     # Số lượng feature tối đa khi lấy sampling
    max_depth=1,        # Độ sâu của cây
    criterion='entropy',# Tiêu chí xây dựng tree
    max_samples=6,      # Tối đa 6 samples cho mỗi cây
    random_state=42     # Đảm bảo kết quả cố định khi chạy lại
)

rf_classifier.fit(x_data, y_data)

# 4. Predict (Cần reshape x_test thành mảng 2D cho đúng định dạng input của sklearn)
x_test = np.array([[2.7, 0.8]])
y_predicted = rf_classifier.predict(x_test)

# In ra kết quả dự đoán (Tên class tương ứng)
predicted_class_name = iris.target_names[y_predicted[0]]
print(f"Mã Class dự đoán: {y_predicted[0]} -> Tên nhãn: {predicted_class_name}")

# 5. Visualize cây đầu tiên (estimators[0])
plt.figure(figsize=(6, 6))
_ = tree.plot_tree(
    rf_classifier.estimators_[0],
    feature_names=['Petal_Length', 'Petal_Width'],
    class_names=iris.target_names,
    filled=True
)
plt.show()
```
#### B. Bài toán Hồi quy (Regression)
* **Số đặc trưng được chọn ngẫu nhiên tại mỗi nút:** $m = \lfloor p / 3 \rfloor$ (giá trị đề xuất)
* **Thước đo điểm phân nhánh:** Sử dụng Variance Reduction hoặc Sum of Squared Errors (SSE).
* **Dự đoán cuối cùng:** Lấy trung bình cộng kết quả của tất cả các cây (**Averaging**):
  $$\hat{Y} = \frac{1}{B} \sum_{b=1}^{B} \hat{y}_b(\mathbf{p}_{new})$$
##### Code Python
```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import matplotlib.pyplot as plt

# 1. Load data từ module datasets của sklearn
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Get data (Lấy 2 đặc trưng: MedInc - Thu nhập trung bình và AveRooms - Số phòng trung bình)
feature_names = ["MedInc", "AveRooms"]
x_data = df[feature_names].to_numpy()
y_data = df["MedHouseVal"].to_numpy()  # Target: Giá nhà trung bình (đơn vị: 100,000$)

# 3. Define Model cho bài toán Hồi quy (Regression)
rf_regressor = RandomForestRegressor(
    n_estimators=3,          # Số lượng decision tree
    max_features=1,          # Số lượng feature tối đa khi lấy sampling tại mỗi node
    max_depth=2,             # Độ sâu của cây (tăng lên 2 để cây có nút phân nhánh rõ hơn)
    criterion='squared_error',# Tiêu chí chia nhánh cho regression (MSE)
    max_samples=6,           # Tối đa 6 samples cho mỗi cây
    random_state=42          # Đảm bảo kết quả cố định khi chạy lại
)

# Huấn luyện mô hình
rf_regressor.fit(x_data, y_data)

# 4. Predict
# x_test gồm: MedInc = 2.7 (Thu nhập ~27,000$) và AveRooms = 4.8 (Số phòng trung bình)
x_test = np.array([[2.7, 4.8]])
y_predicted = rf_regressor.predict(x_test)

print(f"Giá nhà dự đoán: {y_predicted[0]:.4f} ($100,000) -> tương đương ~${y_predicted[0] * 100000:,.2f}")

# 5. Visualize cây đầu tiên (estimators_[0])
plt.figure(figsize=(8, 6))
_ = tree.plot_tree(
    rf_regressor.estimators_[0],
    feature_names=feature_names,
    filled=True
)
plt.title("Decision Tree #1 trong Random Forest Regressor")
plt.show()
```

---
## 5. Fill in missing data with Random Forest

> [!warning]
> Imputation bằng proximity matrix là một kỹ thuật cổ điển liên quan đến Random Forest, nhưng không phải là hành vi mặc định của mọi implementation.
>
> Khi sử dụng thư viện, cần kiểm tra rõ implementation có hỗ trợ proximity matrix và iterative imputation hay không.

Random Forest có thuật toán xử lý dữ liệu khuyết vô cùng mạnh mẽ dựa trên **Proximity Matrix (Ma trận độ tương đồng)**.

![Pasted image 20260723210958](../../../../05_Assets/Images/Pasted%20image%2020260723210958.png)

### Các bước xử lý Missing Value.
#### Bước 1: Guessing the data

* **Số (Numeric):** Thay thế vị trí khuyết bằng giá trị trung vị (Median) của cột đó (hoặc trung vị theo từng Class nếu là bài toán Phân loại). 
* **Phân loại/Chữ (Categorical/Text):** Thay thế bằng giá trị xuất hiện nhiều nhất (Mode) của cột đó.

#### Bước 2: Build Random Forest & Calculate Proximity Matrix for all Tree
* Cho toàn bộ dữ liệu (đã điền tạm) chạy qua Random Forest. 
* **Proximity Matrix $P$:** Là ma trận vuông $N \times N$, trong đó $P_{ij}$ đo mức độ "gần nhau" giữa mẫu $i$ và mẫu $j$. * Nếu mẫu $i$ và mẫu $j$ rơi vào cùng một lá (leaf node) của cây $t$ (hay còn gọi là same decision), giá trị $P_{ij}$ tăng thêm $1$. 
* Sau khi chạy qua $B$ cây, chuẩn hóa: $P_{ij} = \frac{P_{ij}}{B}$.

### Bước 3: Refine the Guesses (Cập nhật lại giá trị đoán)

Dùng Proximity Matrix làm trọng số để ước lượng lại giá trị bị khuyết.
#### Đối với dữ liệu số (Numbering):
Value khuyết của mẫu $i$ tại thuộc tính số $k$ ($\hat{x}_{i, k}$) được tính bằng **Tổng trung bình có trọng số** của tất cả các mẫu không bị khuyết còn lại:
$$\hat{x}_{i, k} = \sum_{j \neq i} \left( x_{j, k} \cdot W_{j \to i} \right)$$

*Trong đó:*
* $W_{j \to i} = \frac{P_{i, j}}{\sum_{m \neq i} P_{i, m}}$ : Trọng số Proximity đã chuẩn hóa của mẫu $j$ đối với mẫu $i$.
* $P_{i, j}$: Độ tương đồng (Proximity) giữa mẫu $i$ và mẫu $j$ trong Proximity Matrix.
* $\sum_{m \neq i} P_{i, m}$: Tổng tất cả các giá trị Proximity trên hàng $i$ với các mẫu không bị khuyết.
#### Các bước thực thi chi tiết (Từng bước tính toán):

1. **Tính trọng số chuẩn hóa ($W_{j \to i}$):** Chia độ tương đồng $P_{i, j}$ cho tổng độ tương đồng của hàng $i$.
2. **Tính giá trị đóng góp thành phần:** Nhạn giá trị thuộc tính $x_{j,k}$ của từng mẫu $j$ với trọng số $W_{j \to i}$ tương ứng.
3. **Tính tổng cuối cùng (Summation):** Cộng toàn bộ các giá trị đóng góp thành phần lại để thu được giá trị ước lượng mới $\hat{x}_{i, k}$:
$$\hat{x}_{i, k} = (x_{1, k} \cdot W_{1 \to i}) + (x_{2, k} \cdot W_{2 \to i}) + \dots + (x_{n, k} \cdot W_{n \to i})$$
<div align="center">
  <img src="Pasted image 20260723213616.png" width="500">
  <img src="Pasted image 20260723220448.png" width="500">
  <img src="Pasted image 20260723220510.png" width="500">
</div>

* **Đối với dữ liệu phân loại/chữ (Text/Categorical):** 
	* Tính Frequency của $class_i$
	* Tính Weight của class (trọng số): $\text{Weight for} class_i = \text{Proximity of} class_i/ \text{all proximities}$.
	* Tính $\text{The weight frequency of} class_i =\text{Frequency of} class_i*\text{Weight for} class_i$
	* Chọn class có $\text{The weight frequency}$ lớn nhất.
<div align="center">
  <img src="Pasted image 20260723213635.png" width="500">
  <img src="Pasted image 20260723213836.png" width="500">
</div>
### Bước 4: Lặp lại

Sau khi cập nhật các giá trị bị thiếu:

1. Huấn luyện lại Random Forest trên dữ liệu mới.
2. Tính lại proximity matrix.
3. Ước lượng lại các giá trị bị thiếu.
4. Lặp lại cho đến khi:
   - thay đổi giữa hai vòng đủ nhỏ; hoặc
   - đạt số vòng lặp tối đa.

Ví dụ điều kiện hội tụ cho numeric feature:
$$
\Delta
=
\frac{
\sum_{(i,k)\in\mathcal M}
\left(
\widehat x_{ik}^{(t)}
-
\widehat x_{ik}^{(t-1)}
\right)^2
}{
\sum_{(i,k)\in\mathcal M}
\left(
\widehat x_{ik}^{(t-1)}
\right)^2
},
$$

trong đó $\mathcal M$ là tập các vị trí ban đầu bị thiếu.
Dừng khi:
$$
\Delta<\epsilon.
$$

## 6. Advantages and Limitations

### Advantages

- Giảm variance tốt hơn một Decision Tree đơn lẻ.
- Hoạt động tốt trên dữ liệu tabular.
- Có thể mô hình hóa quan hệ phi tuyến và feature interactions.
- Không yêu cầu feature scaling.
- Hỗ trợ classification và regression.
- Có thể sử dụng OOB score.
- Có thể đánh giá feature importance.

### Limitations

- Khó giải thích hơn một cây đơn lẻ.
- Tốn bộ nhớ vì lưu nhiều cây.
- Prediction chậm hơn Decision Tree.
- Không ngoại suy tốt trong regression.
- Feature importance dựa trên impurity có thể thiên lệch.
- Có thể gặp khó khăn với dữ liệu rất nhiều chiều hoặc dữ liệu cực kỳ thưa.