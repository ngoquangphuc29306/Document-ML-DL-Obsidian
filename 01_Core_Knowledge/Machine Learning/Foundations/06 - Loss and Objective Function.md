---
title: "Loss Function và Objective Function"
tags: [machine-learning, foundations, loss, objective-function]
status: complete
---
---

# Loss Function và Objective Function

## 1. Loss Function

Loss đo mức độ prediction sai so với target:
$$L(y, \hat{y})$$

*Loss càng nhỏ càng tốt.*

---

## 2. Regression Losses

### Squared Error
$$L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$$

*Phạt lỗi lớn mạnh.*

### Absolute Error
$$L(y, \hat{y}) = |y - \hat{y}|$$

*Ít nhạy với outlier hơn squared error.*

---

## 3. Classification Loss

### Binary Cross-Entropy
$$L(y, p) = - \left[ y\log(p) + (1-y)\log(1-p) \right]$$

Trong đó: $p = P(y=1 \mid x)$.

---

## 4. Empirical Risk

Trung bình loss trên tập huấn luyện (train):
$$R(\theta) = \frac{1}{N} \sum_{i=1}^{N} L(y_i, f_\theta(x_i))$$

---

## 5. Objective Function

Objective là hàm optimizer thực sự tối ưu:
$$J(\theta) = \frac{1}{N} \sum_{i=1}^{N} L(y_i, f_\theta(x_i)) + \lambda\Omega(\theta)$$

Trong đó:
* **Loss:** fit dữ liệu
* **Regularization ($\Omega$):** phạt độ phức tạp (complexity)

---

## 6. Loss khác Metric

Có thể:
* Train bằng **Log Loss**
* Evaluate bằng **Accuracy / F1 / AUC**

*Vì metric dùng cho business evaluation không nhất thiết tối ưu được trực tiếp bằng gradient descent.*

---

## 7. Ví dụ SVM

$$\frac{1}{2}\|w\|^2 + C \sum_{i} \max(0, 1 - y_i f(x_i))$$

* **Hinge loss:** $\max(0, 1 - y_i f(x_i))$ (phạt prediction sai)
* **Regularization:** $\|w\|^2$

---

## 8. Ví dụ XGBoost

$$\text{Objective} = \sum_{i} L(y_i, \hat{y}_i) + \sum_{k} \Omega(f_k)$$

---

> [!NOTE] Summary
> * **Loss:** đo lỗi của từng mẫu dữ liệu.
> * **Objective:** là đại lượng optimizer tối thiểu hóa trên toàn bộ tập dữ liệu, thường gồm **Loss + Regularization**.