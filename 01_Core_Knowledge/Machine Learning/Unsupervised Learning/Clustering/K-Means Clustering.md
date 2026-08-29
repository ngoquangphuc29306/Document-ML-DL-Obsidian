---
tags:
  - unsupervised-learning
  - clustering
  - classification
  - regression
  - ml/algorithm
created: 2026-07-21
---
---

# K-Means Clustering

## 1. What is K-Means Clustering?

* **Clustering**: Là một bài toán thuộc nhóm [[Types of Machine Learning#2. Unsupervised Learning|Unsupervised Learning]].
* **Key Concepts**:
  * **Similarity Metrics**: Gom các data points thành từng nhóm dựa trên thước đo khoảng cách/độ tương đồng.
  * **Grouping**: Mỗi phân cụm (cluster) chứa các sample có tính chất/đặc trưng giống nhau.
* **Applications**:
  * Market Segmentation (Phân đoạn khách hàng)
  * Anomaly Detection (Phát hiện bất thường)
  * Image Segmentation (Phân đoạn hình ảnh)

![K-Means Overview](Pasted%20image%2020260721104056.png)

---

## 2. How K-Means Works

Quy trình hoạt động cơ bản của K-Means gồm **4 bước cốt lõi**:

### Step 1: Chọn tâm cụm ban đầu (Initialize Centroids)
* **Randomly** (hoặc dùng thuật toán tối ưu **K-Means++**) chọn $K$ tâm cụm ($cluster\ centers / centroids$):
  $$\{c_1, c_2, \dots, c_k\}$$

### Step 2: Gán dữ liệu vào cụm gần nhất (Assign Clusters)
* Với mỗi cụm $j$, tạo tập hợp $C_j$ gồm các điểm trong $X$ gần tâm $c_j$ nhất.
* Khoảng cách giữa điểm dữ liệu $x_i$ đến tâm $c_j$ thường tính bằng **Euclidean Distance** (xem chi tiết tại [[Similarity Metrics#1 1 Euclidean Distance Khoảng cách L_2|Similarity Metrics]]):
  $$d(x_i, c_j) = \sqrt{\sum_{l=1}^{m} (x_{il} - c_{jl})^2}$$

### Step 3: Cập nhật tâm cụm mới (Update Centroids)
* Với mỗi cluster $j$, cập nhật lại $c_j$ thành tâm mới bằng cách lấy trung bình cộng tọa độ của tất cả các điểm thuộc cụm đó:
  $$c_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i$$
  *(Trong đó $|C_j|$ là số lượng điểm dữ liệu đang thuộc cụm $C_j$)*

### Step 4: Lặp lại cho đến khi hội tụ (Repeat until Convergence)
* Quay lại **Step 2**, lặp lại quá trình gán cụm và cập nhật tâm cho đến khi thuật toán **hội tụ (converge)**.
* **Dấu hiệu hội tụ**: Vị trí các $centroids$ không còn thay đổi (hoặc sự thay đổi nhỏ hơn một ngưỡng $\epsilon$ cho trước).
![[K-MeanWork.excalidraw|1000]]
---

## 3. Data Normalization (Tiền xử lý)

* **Tại sao cần Normalize?** Do K-Means dựa hoàn toàn vào khoảng cách (Euclidean), nếu các đặc trưng (features) có quy mô (scale) chênh lệch lớn, đặc trưng có giá trị lớn hơn sẽ áp đảo toàn bộ khoảng cách $\rightarrow$ Dẫn đến **Model bị Bias**.
* **Phương pháp phổ biến**: Thường áp dụng **Z-Score Standardization** (xem tại [[Feature Scaling]]).

![Data Normalization Effect](Pasted%20image%2020260721105224.png)

---

---

## 4. Limitations of K-Means (Hạn chế hình học)

K-Means chỉ phân cụm tốt khi các cụm có dạng **hình cầu (spherical)** và **quy mô đồng đều**. Thuật toán sẽ **thất bại** trong 3 trường hợp sau:

1. **Different Sizes (Kích thước khác biệt):** Cụm quá lớn sẽ bị K-Means "nuốt" bớt dữ liệu hoặc chia nhỏ ra để cân bằng với cụm nhỏ.
2. **Different Densities (Mật độ khác biệt):** Cụm quá thưa sẽ bị tâm của cụm dày lấn át, dẫn đến gán sai các điểm dữ liệu ở vùng rìa.
3. **Non-Spherical Shapes (Hình dạng không phải khối cầu):** Không phân cụm được các dạng dữ liệu phức tạp (như hình trăng khuyết, vòng tròn đồng tâm, xoắn ốc) vì K-Means sử dụng [[Similarity Metrics#1 1 Euclidean Distance Khoảng cách L_2|Euclidean Distance]] nên ranh giới phân chia luôn là **đường thẳng/mặt phẳng**.
---

## 5. Optimal K for K-Means (Tìm K tối ưu)

Để chọn số lượng cụm $K$ phù hợp nhất, ta dùng 2 phương pháp phổ biến:

### a) Elbow Method (Phương pháp Cùi chỏ)
Sử dụng chỉ số **WCSS (Within-Cluster Sum of Squares)** - Tổng bình phương khoảng cách từ các điểm tới tâm cụm của chúng:

$$WCSS = \sum_{j=1}^{K} \sum_{x_i \in C_j} \|x_i - c_j\|^2$$

* **Nguyên lý**: Tìm điểm "cùi chỏ" (Elbow point) - nơi mà khi tiếp tục tăng $K$, giá trị $WCSS$ không còn giảm đáng kể nữa.
* **Đặc điểm**: $WCSS$ luôn giảm khi $K$ tăng. Điểm gập của đường cong là $K$ tối ưu.

![Elbow Method Graph](Pasted%20image%2020260721105611.png)

### b) Silhouette Method (Phương pháp Hệ số Silhouette)
Đánh giá chất lượng phân cụm của từng điểm dữ liệu:
* Có **gần** các điểm trong cùng $1$ cluster hay không.
* Có **cách xa** các điểm thuộc cluster khác hay không.

#### Công thức chọn $K$ tối ưu:
$$K^* = \arg\max_{K} \text{Silhouette}(K)$$

#### Công thức tính Hệ số Silhouette $S(i)$:
Giới hạn giá trị: $-1 \le S(i) \le 1$

$$S(i) = \frac{N(i) - C(i)}{\max(N(i), C(i))}$$

Hoặc dạng hàm phân nhánh:
$$S(i) = \begin{cases} 1 - \frac{C(i)}{N(i)}, & \text{if } C(i) < N(i) \\ 0, & \text{if } C(i) = N(i) \\ \frac{N(i)}{C(i)} - 1, & \text{if } C(i) > N(i) \end{cases}$$

**Giải thích biến:**
* $C(i)$ *(hoặc $a(i)$)*: Khoảng cách trung bình từ điểm $i$ đến các điểm khác trong **cùng 1 cluster**.
* $N(i)$ *(hoặc $b(i)$)*: Khoảng cách trung bình nhỏ nhất từ điểm $i$ đến các điểm thuộc **cluster khác gần nhất**.

**Ý nghĩa giá trị $S(i)$:**
* $S(i) \approx 1$: Điểm nằm đúng cluster và cách xa cụm khác (Rất tốt).
* $S(i) \approx 0$: Điểm nằm ngay ranh giới giữa 2 cụm.
* $S(i) < 0$: Điểm có thể đã bị **gán sai cụm**.
