---
tags:
  - ml/algorithm
  - supervised-learning
  - instance-based
  - lazy-learning
  - classification
  - regression
created: 2026-07-20
---

# K-Nearest Neighbors (KNN)

## 1. What is KNN?
**K-Nearest Neighbors (KNN)** là một thuật toán thuộc nhóm [[Types of Machine Learning#1. Supervised Learning|Supervised Learning]]. Thuật toán này có hai đặc điểm cốt lõi:
*   **[[Non-parametric vs Parametric Models|Non-parametric]] (Phi tham số):** Không giả định bất kỳ điều gì về phân phối của dữ liệu nền tảng.
*   **Lazy Learning / Instance-based Learning (Học lười):** Không có quá trình huấn luyện tường minh để tối ưu tham số. Nó chỉ đơn thuần là ghi nhớ toàn bộ dữ liệu huấn luyện. Khi có dữ liệu mới cần dự đoán, nó mới bắt đầu tính toán khoảng cách để đưa ra kết quả.

KNN có thể áp dụng cho cả hai bài toán: [[Types of Machine Learning|Classification]] và [[Types of Machine Learning|Regression]].
![LazyLearning](../../../05_Assets/Excalidraw/LazyLearning.excalidraw.md)

---

## 2. Classification Approach
Chi tiết về định nghĩa bài toán, xem tại: [[Types of Machine Learning|Classification]] .
### Các bước xây dựng quy trình KNN Classification:
1. **Step 1: Look at the data** (Quan sát dữ liệu hiện có).
2. **Step 2: Calculate distance** (Tính khoảng cách từ điểm mới tới tất cả các điểm trong tập Train).
3. **Step 3: Find Neighbors** (Tìm ra $K$ điểm láng giềng gần nhất).
4. **Step 4: Vote labels** (Bỏ phiếu số đông để chọn nhãn).

---

#### Minh họa trực quan từng bước:

**Step 1: Look at the data**

*Train Data:*

| Petal Length | Petal Width | Label |
| :----------- | :---------- | :---- |
| 1.4          | 0.2         | 0     |
| 1.3          | 0.4         | 0     |
| 4.0          | 1.0         | 1     |
| 4.7          | 1.4         | 1     |

*Test Data (Điểm cần dự đoán):* $X_{test} = (2.4, 0.8)$

**Step 2: Calculate Distance (Sử dụng Euclidean Distance)**

Công thức: $d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$

| Petal Length | Petal Width | Label | Distance to Test Point |
| :----------- | :---------- | :---- | :--------------------- |
| 1.4          | 0.2         | 0     | **1.166**              |
| 1.3          | 0.4         | 0     | **1.170**              |
| 4.0          | 1.0         | 1     | **1.612**              |
| 4.7          | 1.4         | 1     | 2.377                  |

**Step 3: Find Neighbors (Sắp xếp tăng dần theo khoảng cách)**

1. 1st: $(1.4, 0.2)$
2. 2nd: $(1.3, 0.4)$
3. 3rd: $(4.0, 1.0)$
4. 4th: $(4.7, 1.4)$

**Step 4: Vote labels**

Chọn hệ số $K = 3 \rightarrow$ Lấy 3 điểm láng giềng đầu tiên.
* Trong 3 điểm này: có **2 điểm nhãn 0** và **1 điểm nhãn 1**.
* Vì số lượng `Class 0 > Class 1` nên nhãn dự đoán cuối cùng là: $\hat{y} = 0$.

---

## 3. Regression Approach
Chi tiết về định nghĩa bài toán, xem tại: [[Types of Machine Learning|Regression]] .

### Quy trình thực hiện:
*   **Step 1, Step 2, Step 3:** Hoàn toàn tương tự bài toán Classification.
*   **Step 4: Compute the mean value (Tính giá trị trung bình):** Giá trị dự đoán $\hat{y}$ sẽ là trung bình cộng giá trị của $K$ láng giềng gần nhất.
    $$\hat{y} = Y_{pred} = \frac{1}{K} \sum_{x \in Neighbors} y_x$$

---

## 4. Geometry Distance Functions

Tùy thuộc vào đặc tính không gian dữ liệu, ta chọn hàm tính khoảng cách thích hợp:
Tham khảo chi tiết tại [[Similarity Metrics]].
*   **Euclidean Distance ($p = 2$):** Khoảng cách đường thẳng hình học thông thường.
    $$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$
*   **Manhattan Distance ($p = 1$):** Khoảng cách đi theo các đường vuông góc (bàn cờ).
    $$d(x, y) = \sum_{i=1}^{n} \vert{}x_i - y_i\vert{}$$
*   **Minkowski Distance ($p = \text{norm}$):** Dạng tổng quát hóa của Euclidean và Manhattan.
    $$d(x, y) = \left( \sum_{i=1}^{n} \vert{}x_i - y_i\vert{}^p \right)^{\frac{1}{p}}$$
*   **Chebyshev Distance ($p \to \infty$):** Khoảng cách lớn nhất trên một chiều đơn lẻ duy nhất.
    $$d(x, y) = \max_{i} (\vert{}x_i - y_i\vert{}) = \lim_{p \to \infty} \left( \sum_{i=1}^{n} \vert{}x_i - y_i\vert{}^p \right)^{\frac{1}{p}}$$

---

## 5. Feature Scaling (Chuẩn hóa dữ liệu)
> **Lưu ý cốt lõi:** Khoảng cách hình học sẽ bị ảnh hưởng nặng nề bởi các đặc trưng có phạm vi giá trị (range) lớn hơn, khiến mô hình bị phụ thuộc vào chỉ một feature đó (Ví dụ: 1 feature ở đơn vị `cm`, 1 feature ở đơn vị `mm`).

Tham khảo qua : [[Feature Scaling]]
Giải pháp là đưa các dải giá trị về cùng một quy chuẩn thông qua bước Preprocessing trong [[Machine Learning Workflow#Bước 2 Preprocessing Cleaning|ML Workflow]]:

*   **MinMaxScaler:** Biến đổi dữ liệu về đoạn $[0, 1]$.
    $$x_{new} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
*   **Z-Score (StandardScaler):** Biến đổi dữ liệu về phân phối có trung bình $\mu = 0$ và độ lệch chuẩn $\sigma = 1$.
    $$x_{new} = \frac{x - \mu}{\sigma}$$
---

## 6. Searching in KNN (Thuật toán tìm kiếm láng giềng)

Giả sử tập huấn luyện có:

- $N$: số mẫu;
- $D$: số chiều của dữ liệu;
- $K$: số láng giềng cần tìm.

Độ phức tạp của KNN cần được tách thành hai giai đoạn:

1. **Xây dựng cấu trúc tìm kiếm**;
2. **Truy vấn một điểm mới**.

### 6.1. Brute Force

Brute Force không xây dựng cấu trúc cây. Khi cần dự đoán một điểm mới, thuật toán tính khoảng cách từ điểm đó đến toàn bộ $N$ mẫu huấn luyện.

#### Chi phí xây dựng

Gần như không có bước xây dựng đáng kể:

$$
\mathcal{O}(1)
$$

nếu chỉ lưu dữ liệu huấn luyện.

#### Chi phí truy vấn một điểm

Tính khoảng cách đến toàn bộ $N$ mẫu, mỗi khoảng cách cần xét $D$ chiều:

$$
\mathcal{O}(ND).
$$

Sau đó cần chọn $K$ khoảng cách nhỏ nhất:

- nếu sắp xếp toàn bộ:

$$
\mathcal{O}(N\log N);
$$

- nếu dùng thuật toán chọn hoặc heap phù hợp:

$$
\mathcal{O}(N\log K).
$$

Vì vậy, chi phí dự đoán một điểm thường có thể mô tả gần đúng là:

$$
\mathcal{O}(ND + N\log K).
$$

Trong nhiều trường hợp, phần tính khoảng cách $\mathcal{O}(ND)$ là thành phần chi phối.

> [!NOTE]
> Công thức $\mathcal{O}(DN^2)$ không phải là độ phức tạp của một lần truy vấn KNN.  
> Nó chỉ có thể xuất hiện nếu tính khoảng cách giữa mọi cặp mẫu trong toàn bộ tập dữ liệu.

---

### 6.2. K-D Tree

K-D Tree chia không gian dữ liệu theo từng trục đặc trưng để loại bỏ các vùng không cần tìm kiếm.

#### Chi phí xây dựng

Trong trường hợp điển hình:

$$
\mathcal{O}(ND\log N).
$$

#### Chi phí truy vấn

Trong điều kiện thuận lợi, đặc biệt khi số chiều nhỏ và dữ liệu phân bố tương đối tốt:

$$
\mathcal{O}(D\log N)
$$

cho một truy vấn gần đúng về mặt trực giác.

Tuy nhiên, đây không phải bảo đảm trong mọi trường hợp. Ở trường hợp xấu nhất, thuật toán có thể phải duyệt gần như toàn bộ dữ liệu:

$$
\mathcal{O}(ND).
$$

K-D Tree thường hoạt động tốt khi:

- số chiều nhỏ;
- dữ liệu có cấu trúc không gian rõ ràng;
- các điểm không quá tập trung hoặc phân bố quá bất thường.

---

### 6.3. Ball Tree

Ball Tree tổ chức dữ liệu thành các vùng hình cầu lồng nhau. Mỗi nút biểu diễn một nhóm điểm nằm trong một vùng có tâm và bán kính.

#### Chi phí xây dựng

Thông thường:

$$
\mathcal{O}(ND\log N).
$$

#### Chi phí truy vấn

Hiệu năng phụ thuộc mạnh vào:

- số chiều;
- phân bố dữ liệu;
- loại metric;
- mức độ tách biệt giữa các cụm điểm.

Trong trường hợp thuận lợi, Ball Tree có thể loại bỏ nhiều vùng không liên quan và truy vấn nhanh hơn Brute Force. Tuy nhiên, trong trường hợp xấu nhất:

$$
\mathcal{O}(ND).
$$

Ball Tree thường linh hoạt hơn K-D Tree đối với:

- một số metric không hoàn toàn phù hợp với việc chia theo trục;
- dữ liệu có cấu trúc cụm;
- số chiều trung bình.

---

### 6.4. So sánh tổng quát

| Phương pháp | Chi phí xây dựng | Chi phí truy vấn một điểm | Điều kiện phù hợp |
|---|---:|---:|---|
| Brute Force | Gần $\mathcal{O}(1)$ | Khoảng $\mathcal{O}(ND)$ | Dữ liệu nhỏ hoặc số chiều rất lớn |
| K-D Tree | Khoảng $\mathcal{O}(ND\log N)$ | Tốt nhất gần $\mathcal{O}(D\log N)$, xấu nhất $\mathcal{O}(ND)$ | Số chiều thấp |
| Ball Tree | Khoảng $\mathcal{O}(ND\log N)$ | Phụ thuộc dữ liệu, xấu nhất $\mathcal{O}(ND)$ | Số chiều trung bình, dữ liệu có cấu trúc cụm |

> [!IMPORTANT]
> Không có cấu trúc tìm kiếm nào luôn nhanh nhất.  
> Khi số chiều tăng cao, cả K-D Tree và Ball Tree đều có thể mất lợi thế do hiện tượng **curse of dimensionality**.

---

## 7. Find Optimal K
$K$ là một siêu tham số (**Hyperparameter**) ảnh hưởng trực tiếp đến kết quả dự đoán:
*   **K quá lớn:** Đường ranh giới phân loại quá phẳng (smooth), không bắt được xu hướng thực tế của dữ liệu $\rightarrow$ Dẫn đến hiện tượng [[Overfitting vs Underfitting#Underfitting Học vẹt chưa tới Quá đơn giản|Underfitting]].
*   **K quá nhỏ:** Mô hình nhạy cảm quá mức với các điểm nhiễu (outliers) $\rightarrow$ Dẫn đến hiện tượng [[Overfitting vs Underfitting#Overfitting Học vẹt quá đà Quá phức tạp|Overfitting]].

### Giải pháp xử lý khi chọn K chẵn (Tránh hòa phiếu):
Khi số lượng phiếu bầu cho các nhóm bằng nhau, ta áp dụng các phương pháp gán trọng số (**weights**):
1.  **Uniform weight:** Mọi láng giềng đều có trọng số bình đẳng (mặc định).
2.  **Distance weight ($w = \frac{1}{d}$):** Trọng số tỉ lệ nghịch với khoảng cách. Điểm nào đứng gần điểm cần dự đoán hơn sẽ có trọng lượng phiếu bầu lớn hơn.
3.  **User-defined weight:** Trọng số tự định nghĩa dựa trên bài toán cụ thể.
---

## 8. Đánh giá và Chọn K dựa trên Validation Set
Quá trình chọn $K$ tối ưu được thực hiện bằng cách đánh giá hiệu suất mô hình trên tập kiểm thử (Validation set) thông qua các chỉ số đo lường. 

Quá trình chọn **K** có thể áp dụng phương pháp [[Cross-Validation]] để tìm. 

Chi tiết định nghĩa, công thức và bài toán áp dụng của từng chỉ số được lưu trữ tại:
*   [[../Evaluation/Classifier Metric#Accuracy Độ chính xác tổng quan|Accuracy]]
*   [[../Evaluation/Classifier Metric#Precision Độ chính xác của lựa chọn|Precision]]
*   [[../Evaluation/Classifier Metric#Recall Sensitivity Độ phủ Tỷ lệ tìm sót|Recall]]
*   [[../Evaluation/Classifier Metric#F1-Score|F1-Score]]

![Pasted image 20260720203856](../../../05_Assets/Images/Pasted%20image%2020260720203856.png)

![Pasted image 20260721072800](../../../05_Assets/Images/Pasted%20image%2020260721072800.png)

![Pasted image 20260721072818](../../../05_Assets/Images/Pasted%20image%2020260721072818.png)

---

## 9. Triển khai KNN với Scikit-Learn (sklearn)

### 9.1. KNN Classification
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Giả lập dữ liệu mẫu
X = np.array([[1.4, 0.2], [1.3, 0.4], [4.0, 1.0], [4.7, 1.4]])
y = np.array([0, 0, 1, 1])
X_new = np.array([[2.4, 0.8]])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_new_scaled = scaler.transform(X_new)

# Khởi tạo mô hình Classifier với K = 3
knn_clf = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='minkowski', p=2)
knn_clf.fit(X_scaled, y)

# Dự đoán dữ liệu mới
prediction = knn_clf.predict(X_new_scaled)
print(f"Nhãn dự đoán: {prediction[0]}")
```

### 9.2. KNN Regression
```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# 1. Tạo dữ liệu từ bảng trong ảnh
data = {
    'Experience': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
    'Salary': [0.0, 0.0, 0.0, 0.0, 60.0, 64.0, 55.0, 61.0, 66.0, 83.0, 93.0, 91.0, 98.0, 101.0]
}

df = pd.DataFrame(data)

# 2. Tách biến đầu vào (X) và biến mục tiêu (y)
X = df[['Experience']]  # Cần ở dạng 2D array / DataFrame
y = df['Salary']

# 3. Khởi tạo và huấn luyện mô hình KNN Regressor (ví dụ k = 3)
k = 3
knn_reg = KNeighborsRegressor(n_neighbors=k, weights='distance',
		   metric='euclidean')
knn_reg.fit(X, y)

# 4. Dự đoán lương dựa trên kinh nghiệm mới (ví dụ: Experience = 4.2 năm)
X_new = np.array([[4.2]])
predicted_salary = knn_reg.predict(X_new)

print(f"Mô hình KNN Regressor (k={k})")
print(f"Lương dự đoán cho {X_new[0][0]} năm kinh nghiệm là: {predicted_salary[0]:.2f}")
```

### 9.3 KNN Find K with K-Fold
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# 1. Nạp và Chuẩn hóa dữ liệu đơn giản
iris = load_iris()
X = StandardScaler().fit_transform(iris.data) # Chuẩn hóa đặc trưng
y = iris.target
  
# 2. Thử nghiệm các giá trị K từ 1 đến 30
k_range = range(1, 10)
k_scores = []

for k in k_range:
    # Khởi tạo KNN với K tương ứng
    knn = KNeighborsClassifier(n_neighbors=k)
    # Chạy 5-Fold Cross-Validation (cv=5 chia dữ liệu thành 5 phần)
    # Hàm tự động train trên 4 phần, test trên 1 phần và lặp lại 5 lần
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    # Lấy điểm trung bình của 5 lần chạy
    k_scores.append(scores.mean())

# 3. Tìm K tốt nhất
best_k = k_range[np.argmax(k_scores)]
print(f"-> Giá trị K tốt nhất là: K = {best_k} với Accuracy = {max(k_scores)*100:.2f}%")

# 4. Vẽ đồ thị
plt.figure(figsize=(9, 5))
plt.plot(k_range, k_scores, color='blue', linestyle='--', marker='o', markerfacecolor='red')
plt.title('Độ chính xác (Accuracy) theo từng giá trị K')
plt.xlabel('Giá trị K (n_neighbors)')
plt.ylabel('Validation Accuracy (Trung bình 5-Fold)')
plt.grid(True)
plt.show()
```