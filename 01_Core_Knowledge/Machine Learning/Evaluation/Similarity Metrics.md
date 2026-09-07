---
tags:
  - math
  - metrics
  - similarity
  - ml/evaluation
created: 2026-07-21
---

# Similarity Metrics (Các thước đo độ tương đồng)

Trong Machine Learning, **Similarity Metric** (Thước đo độ tương đồng) hoặc **Distance Metric** (Thước đo khoảng cách) là công cụ toán học dùng để định lượng mức độ "giống nhau" hoặc "khác nhau" giữa hai điểm dữ liệu/vector $u$ và $v$.

> **Quy tắc cơ bản:**
> * **Distance Metric (Khoảng cách):** Giá trị càng **nhỏ** $\rightarrow$ Hai vector càng **giống** nhau ($d \ge 0$).
> * **Similarity Metric (Độ tương đồng):** Giá trị càng **lớn** $\rightarrow$ Hai vector càng **giống** nhau (thường chuẩn hóa về $[-1, 1]$ hoặc $[0, 1]$).

---

## 1. Nhóm thước đo khoảng cách không gian (Spatial / Geometric Distances)

Dùng cho dữ liệu số liên tục (Continuous Numerical Features).

### 1.1. Euclidean Distance (Khoảng cách $L_2$)
* **Bản chất:** Khoảng cách đường thẳng ngắn nhất nối liền 2 điểm trong không gian $n$ chiều.
* **Công thức:**
  $$d(u, v) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$$
* **Ứng dụng:** Mặc định trong các bài toán hình học, [[K-Nearest Neighbors]], K-Means Clustering.
* **Nhược điểm:** Cực kỳ nhạy cảm với dải quy mô (scale) của các tính năng. Bắt buộc phải áp dụng [[K-Nearest Neighbors#5 Feature Scaling Chuẩn hóa dữ liệu|Feature Scaling]] trước khi tính.

### 1.2. Manhattan Distance (Khoảng cách $L_1$ / City Block)
* **Bản chất:** Tổng khoảng cách dịch chuyển tuyệt đối theo các trục tọa độ vuông góc (như cách di chuyển giữa các ô bàn cờ hay khối nhà thành phố).
* **Công thức:**
  $$d(u, v) = \sum_{i=1}^{n} |u_i - v_i|$$
* **Ứng dụng:** Tốt hơn Euclidean khi dữ liệu có số chiều cao (High-dimensional data) hoặc dữ liệu dạng lưới (Grid-based / Robotics path planning).

### 1.3. Minkowski Distance (Tổng quát hóa $L_p$)
* **Bản chất:** Khung tổng quát cho cả Euclidean ($p=2$) và Manhattan ($p=1$).
* **Công thức:**
  $$d(u, v) = \left( \sum_{i=1}^{n} |u_i - v_i|^p \right)^{\frac{1}{p}}$$

### 1.4. Chebyshev Distance ($L_\infty$)
* **Bản chất:** Khoảng cách bằng độ chênh lệch tuyệt đối lớn nhất trên duy nhất một chiều không gian.
* **Công thức:**
  $$d(u, v) = \max_{i} (|u_i - v_i|)$$

---

## 2. Nhóm thước đo góc và hướng (Angular / Directional Metrics)

Được dùng khi **hướng** của vector quan trọng hơn **độ dài (magnitude)** của nó.

### 2.1. Cosine Similarity (Độ tương đồng Cosine)
* **Bản chất:** Đo góc $\theta$ tạo bởi hai vector trong không gian. Không bị ảnh hưởng bởi độ dài/quy mô của vector.
* **Công thức:**
  $$\text{Cosine Similarity}(u, v) = \cos(\theta) = \frac{u \cdot v}{\|u\| \|v\|} = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \cdot \sqrt{\sum_{i=1}^{n} v_i^2}}$$
* **Dải giá trị:** $[-1, 1]$
  * $1$: Hai vector cùng hướng hoàn toàn ($\theta = 0^\circ$).
  * $0$: Hai vector vuông góc, không liên quan ($\theta = 90^\circ$).
  * $-1$: Hai vector ngược hướng hoàn toàn ($\theta = 180^\circ$).
* **Khoảng cách Cosine (Cosine Distance):**
  $$\text{Cosine Distance} = 1 - \text{Cosine Similarity}$$
* **Ứng dụng:** 
  * Xử lý ngôn ngữ tự nhiên (NLP): So sánh độ tương đồng giữa hai văn bản, so sánh các vector nhúng từ (Word Embeddings / Document Embeddings).
  * Recommender Systems (Hệ thống gợi ý sản phẩm).

### 2.2. Pearson Correlation Coefficient (Hệ số tương quan Pearson)
* **Bản chất:** Thực chất là Cosine Similarity trên các vector đã được trừ đi giá trị trung bình ($\mu$). Đo mối quan hệ tuyến tính giữa hai biến.
* **Công thức:**
  $$r = \frac{\sum_{i=1}^{n} (u_i - \bar{u})(v_i - \bar{v})}{\sqrt{\sum_{i=1}^{n} (u_i - \bar{u})^2} \sqrt{\sum_{i=1}^{n} (v_i - \bar{v})^2}}$$
* **Dải giá trị:** $[-1, 1]$.

---

## 3. Nhóm thước đo trên tập hợp & Chuỗi văn bản (Set & String Metrics)

Dùng cho dữ liệu rời rạc, phân loại (Categorical Data), chuỗi văn bản (Text strings) hoặc tập hợp.

### 3.1. Jaccard Similarity (Độ tương đồng Jaccard)
* **Bản chất:** Tỷ lệ giữa kích thước tập giao (phần chung) và tập hợp (tổng thể) của hai tập hợp $A$ và $B$.
* **Công thức:**
  $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
* **Khoảng cách Jaccard (Jaccard Distance):**
  $$d_J(A, B) = 1 - J(A, B)$$
* **Ứng dụng:** So sánh độ tương đồng về mặt từ ngữ giữa 2 văn bản (BoW - Bag of Words), phát hiện đạo văn (Plagiarism detection).

### 3.2. Hamming Distance (Khoảng cách Hamming)
* **Bản chất:** Đếm số vị trí mà ký tự/phần tử tại đó **khác nhau** giữa hai chuỗi có **cùng độ dài**.
* **Ví dụ:**
  * `u = "karolin"`
  * `v = "kathrin"`
  * Vị trí khác nhau: 'r'-'t', 'o'-'h' $\rightarrow$ $d_{\text{Hamming}} = 2$.
* **Ứng dụng:** Mạng truyền thông (sửa lỗi code / error detection), so sánh chuỗi DNA, kiểm tra mã nhị phân.

---

## 4. Bảng tổng hợp & Hướng dẫn lựa chọn trong thực tế

| Thước đo (Metric) | Loại dữ liệu phù hợp | Ưu điểm chính | Nhược điểm / Lưu ý |
| :--- | :--- | :--- | :--- |
| **Euclidean** | Dữ liệu số liên tục, số chiều thấp | Tự nhiên, dễ hình dung theo hình học | Bị ảnh hưởng mạnh bởi Feature Scaling & "Cờn ác mộng số chiều" (Curse of Dimensionality). |
| **Manhattan** | Dữ liệu số, không gian lưới, số chiều cao | Ít bị ảnh hưởng bởi điểm dị biệt (outliers) hơn Euclidean | Chỉ di chuyển theo trục tọa độ. |
| **Cosine** | Vector thưa (Sparse vectors), văn bản, nhúng AI | Không phụ thuộc vào độ dài/quy mô vector | Không tính đến sự chênh lệch về độ lớn tuyệt đối giữa 2 điểm. |
| **Jaccard** | Dữ liệu tập hợp, từ ngữ | Hoạt động tốt trên tập dữ liệu dạng Bật/Tắt (Binary/Boolean) | Không quan tâm đến tần suất xuất hiện của phần tử. |
| **Hamming** | Chuỗi ký tự, mã Nhị phân | Tính toán cực nhanh | Hai chuỗi bắt buộc phải có độ dài bằng nhau. |

---