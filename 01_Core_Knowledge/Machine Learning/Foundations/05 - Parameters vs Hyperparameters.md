---
title: "Parameters vs Hyperparameters"
tags: [machine-learning, foundations, parameters, hyperparameters]
status: complete
---
---
# Parameters vs Hyperparameters

## 1. Parameter

Parameter là giá trị model **tự học từ training data**.

**Ví dụ Linear Regression:**
$$\hat{y} = w_1 x_1 + \cdots + w_d x_d + b$$

*Các $w_j$ và $b$ là parameters.*

---

## 2. Hyperparameter

Hyperparameter là cấu hình điều khiển quá trình học hoặc độ phức tạp của model.

**Ví dụ:**
* **Decision Tree:** `max_depth`
* **Random Forest:** `n_estimators`
* **SVM:** `C`, `gamma`
* **XGBoost:** `learning_rate`, `max_depth`, `reg_lambda`

---

## 3. So sánh

| Tiêu chí | Parameter | Hyperparameter |
| :--- | :--- | :--- |
| **Ai xác định?** | Model tự học | Người dùng / Tuning |
| **Học trực tiếp từ train?** | Có | Không |
| **Ví dụ** | weights, bias, leaf value | C, gamma, depth |

---
## 4. Training

$$\theta^* = \arg\min_\theta J(\theta)$$

Model tìm parameters $\theta$ tối ưu.

---
## 5. Tuning

$$\lambda^* = \arg\max_{\lambda \in \Lambda} \text{Score}_{\text{validation}}(\lambda)$$

Hyperparameters được chọn dựa trên Validation / Cross-Validation (CV).

---

## 6. Ví dụ XGBoost

### Parameters
* Split thresholds
* Tree structure
* Leaf weights

### Hyperparameters
* `max_depth`
* `learning_rate`
* `n_estimators`
* `gamma`

---

> [!important]
> Hyperparameter không phải “parameter khó hơn”; chúng khác nhau ở **cách được xác định**.