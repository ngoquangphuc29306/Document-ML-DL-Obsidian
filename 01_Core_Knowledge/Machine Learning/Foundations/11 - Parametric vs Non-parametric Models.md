---
title: "Parametric vs Non-parametric Models"
aliases: ["Mô hình tham số và phi tham số"]
tags: [machine-learning, foundations, model-complexity]
status: complete
---
---

# Parametric vs Non-parametric Models

## Ý tưởng cốt lõi

* **Non-parametric** không có nghĩa là mô hình không có parameter.
* **Điểm khác biệt quan trọng:** Cách độ phức tạp (complexity) của mô hình bị giới hạn khi dung lượng dữ liệu huấn luyện tăng lên.

---

## 1. Parametric Models

Một **parametric model** giả định một dạng mô hình cố định được mô tả bởi một số hữu hạn các parameters:

$$f_\theta(x)$$

với:
$$\theta = (\theta_1, \ldots, \theta_p)$$

> [!NOTE] Đặc điểm cốt lõi
> Nếu dạng mô hình (model form) cố định, số lượng parameter ($p$) **không tăng** chỉ vì có thêm các mẫu dữ liệu huấn luyện (training samples).

### Ví dụ: Linear Regression
$$\hat{y} = w_0 + w_1 x_1 + \cdots + w_d x_d$$

Nếu số lượng đặc trưng $d$ cố định, mô hình luôn có đúng:
$$d + 1$$
parameters. Dù tập dữ liệu có $1,000$ hay $1,000,000$ dòng thì số lượng hệ số (coefficients) cơ bản vẫn giữ nguyên.

### Các ví dụ Parametric phổ biến:
* Linear Regression
* Logistic Regression
* Naive Bayes (với phân phối xác định)
* Linear Discriminant Analysis (LDA)

### Ưu & Nhược điểm
* **Ưu điểm:**
  * Đơn giản, trực quan
  * Tốc độ huấn luyện và suy luận (inference) nhanh
  * Cần ít dữ liệu hơn nếu các giả định (assumptions) phù hợp
  * Dễ giải thích (interpretability)
* **Nhược điểm:**
  * Mang các giả định mạnh
  * Không không gian giả thuyết (hypothesis space) bị hạn chế
  * Dễ bị **Underfit** khi mối quan hệ thực tế quá phức tạp

---

## 2. Non-parametric Models

**Non-parametric model** không bị giới hạn bởi một bộ tham số hữu hạn cố định. Độ phức tạp hiệu dụng của mô hình **có thể tăng lên khi lượng dữ liệu tăng**.

### K-Nearest Neighbors (KNN)
* KNN lưu trữ toàn bộ các mẫu huấn luyện.
* **Dự đoán:** Tính khoảng cách $\rightarrow$ Tìm $K$ điểm gần nhất $\rightarrow$ Bầu chọn (vote) hoặc lấy trung bình (average).
* *Dữ liệu tăng $\rightarrow$ Số điểm cần lưu tăng $\rightarrow$ Cấu trúc ranh giới quyết định (decision boundary) phức tạp hơn.*

### Decision Tree
Decision Tree được xếp vào nhóm non-parametric vì số lượng nút/lá (node/leaf) có thể tăng theo dữ liệu và không bị cố định trước bởi một dạng hàm số hữu hạn.
* **Kiểm soát độ phức tạp (Regularization):** `max_depth`, `min_samples_leaf`, cắt tỉa (pruning).

### Random Forest
Là tập hợp (ensemble) của nhiều cây quyết định. Thuộc nhóm non-parametric vì hàm quyết định có độ linh hoạt rất lớn và không cố định số chiều tham số như các mô hình tuyến tính.

### Kernel Methods (e.g., Kernel SVM)
Được xem là non-parametric trong thống kê vì số lượng vectơ hỗ trợ (support vectors) có thể tăng lên cùng với kích thước tập dữ liệu:

$$f(x) = \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b$$

---

## 3. Bảng so sánh

| Tiêu chí | Parametric | Non-parametric |
| :--- | :--- | :--- |
| **Functional form** | Giả định rõ ràng, cố định | Linh hoạt, không giả định dạng hàm |
| **Complexity** | Tương đối cố định | Có thể tăng theo dữ liệu |
| **Bias** | Thường cao hơn | Thường thấp hơn (nếu đủ dữ liệu) |
| **Variance** | Thường thấp hơn | Có thể cao hơn |
| **Data requirement** | Yêu cầu ít dữ liệu hơn | Cần nhiều dữ liệu hơn |
| **Interpretability** | Dễ giải thích hơn | Khó giải thích hơn |
| **Memory / Compute** | Thấp hơn | Có thể rất cao khi dữ liệu lớn |

> [!WARNING] Lưu ý
> Bảng so sánh trên mô tả xu hướng tổng quát, không phải là luật tuyệt đối trong mọi tình huống.

---

## 4. Ví dụ trực giác

Giả sử mối quan hệ thực tế có dạng:
$$y = \sin(x) + \epsilon$$

* **Linear Regression:** $\hat{y} = w_0 + w_1 x \rightarrow$ Chỉ biểu diễn được đường thẳng, rất dễ bị underfit vì biểu thức thực tế có độ cong.
* **KNN Regression:** Không giả định đường thẳng; sử dụng các mẫu dữ liệu lân cận $\rightarrow$ Rất linh hoạt và bám sát đường cong $\sin(x)$ nếu có đủ dữ liệu.

---

## 5. Liên hệ Bias–Variance

### Parametric
* Không gian giả thuyết (hypothesis space) nhỏ hơn.
* **Bias** thường cao hơn (High Bias).
* **Variance** thường thấp hơn (Low Variance).

### Non-parametric
* Không gian giả thuyết linh hoạt hơn.
* **Bias** có thể thấp hơn (Low Bias).
* **Variance** có thể cao hơn (High Variance / Nguy cơ Overfitting).

---

## 6. Những hiểu nhầm cần tránh

### ❌ "Non-parametric = Không học parameter"
> **Sai.** Decision Tree vẫn học các tham số cụ thể: *split feature*, *threshold*, và *leaf output*.

### ❌ "Non-parametric = Lazy Learning"
> **Sai.** 
> * **KNN:** Vừa là *non-parametric*, vừa là *lazy learning* (chỉ tính toán khi dự đoán).
> * **Decision Tree:** Là *non-parametric*, nhưng là *eager learning* (huấn luyện xong mô hình trước khi dự đoán).

$$\text{Non-parametric} \neq \text{Lazy Learning}$$

---

## 7. Khi nào ưu tiên Parametric?

* Tập dữ liệu nhỏ.
* Bắt buộc cần tính giải thích cao (Interpretability).
* Cần tốc độ suy luận (inference) cực nhanh.
* Mối quan hệ giữa dữ liệu và nhãn tương đối đơn giản.
* Muốn xây dựng một mô hình cơ sở (Baseline) nhanh chóng.

---

## 8. Khi nào ưu tiên Non-parametric?

* Dữ liệu huấn luyện đủ nhiều.
* Mối quan hệ giữa đặc trưng và mục tiêu phức tạp, phi tuyến.
* Cần mô hình có độ linh hoạt tối đa.
* Không muốn đưa ra giả định mạnh về dạng hàm toán học của dữ liệu (functional form).

---

> [!TIP] Ghi nhớ cốt lõi
> * **Parametric:** Độ phức tạp được mô tả bởi một bộ parameters hữu hạn cố định.
> * **Non-parametric:** Độ phức tạp có thể tăng cùng dữ liệu và linh hoạt hơn về dạng hàm (functional form).