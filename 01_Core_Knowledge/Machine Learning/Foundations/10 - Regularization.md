---
title: "Regularization"
tags: [machine-learning, foundations, regularization]
status: complete
---
---
# Regularization

## 1. Khái niệm

Regularization là tập hợp các kỹ thuật hạn chế mô hình trở nên quá phức tạp nhằm cải thiện khả năng khái quát hóa (generalization).

$$\text{Objective} = \text{Training Loss} + \text{Complexity Penalty}$$

---

## 2. L2 Regularization (Ridge)

$$\Omega(w) = \sum_j w_j^2$$

**Hàm mục tiêu (Objective):**
$$J(w) = \text{Loss} + \lambda \sum_j w_j^2$$

**Tác dụng:**
* Thu nhỏ các hệ số (shrink coefficients)
* Hạn chế các trọng số (weights) có giá trị cực lớn
* Thường giúp giảm Variance (độ biến động)

*Ridge Regression sử dụng L2 Regularization.*

---

## 3. L1 Regularization (Lasso)

$$\Omega(w) = \sum_j |w_j|$$

**Tác dụng:**
* Thu nhỏ các trọng số
* Có khả năng triệt tiêu hẳn một số hệ số về $0$ (tự động chọn lọc đặc trưng)
* Tạo mô hình thưa thớt (sparse model)

*Lasso Regression sử dụng L1 Regularization.*

---

## 4. Vai trò của siêu tham số $\lambda$

* **$\lambda$ nhỏ:** Regularization yếu $\rightarrow$ mô hình tự do hơn để fit tập train.
* **$\lambda$ lớn:** Regularization mạnh $\rightarrow$ ép mô hình đơn giản hơn.
* **$\lambda$ quá lớn:** Có thể khiến mô hình bị **Underfitting**.

---

## 5. Support Vector Machine (SVM)

Tham số $C$ điều khiển sự đánh đổi (tradeoff) giữa khoảng cách lề (margin) và vi phạm (violation):

* **$C$ lớn:** Phạt lỗi vi phạm mạnh $\rightarrow$ Regularization yếu hơn (khớp chặt tập train).
* **$C$ nhỏ:** Chấp nhận nhiều vi phạm hơn $\rightarrow$ Regularization mạnh hơn (lề rộng hơn).

---

## 6. Decision Tree

Regularization thông qua các ràng buộc cấu trúc:
* `max_depth` (độ sâu tối đa)
* `min_samples_leaf` (số mẫu tối thiểu ở lá)
* `pruning` (cắt tỉa cây)

---

## 7. Random Forest

Mô hình giảm Variance nhờ phối hợp:
* Bootstrap sampling (lấy mẫu có hoàn lại)
* Random feature subsets (chọn ngẫu nhiên tập con đặc trưng)
* Averaging trees (lấy trung bình kết quả các cây)

---

## 8. XGBoost

Phạt độ phức tạp của từng cây theo công thức:

$$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_j w_j^2$$

Trong đó:
* $\gamma$: Phạt số lượng lá $T$ của cây (tree complexity)
* $\lambda$: L2 regularization trên trọng số lá $w_j$

---

## 9. Early Stopping

Việc dừng quá trình huấn luyện trước khi mô hình bắt đầu học thuộc lòng nhiễu (fit noise) cũng đóng vai trò như một kỹ thuật Regularization hiệu quả.

---

> [!important]
> Regularization không nhằm mục đích làm cho train error đạt mức thấp nhất; mục tiêu chính của nó là **tối ưu hóa hiệu năng trên dữ liệu chưa từng thấy (unseen data)**.