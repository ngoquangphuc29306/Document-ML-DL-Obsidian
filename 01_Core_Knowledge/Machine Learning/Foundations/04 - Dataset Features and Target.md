---
title: "Dataset, Features và Target"
tags: [machine-learning, foundations, dataset]
status: complete
---
# Dataset, Features và Target

> [!abstract]
> Đây là bộ ký hiệu và khái niệm cơ bản để đọc hầu hết tài liệu Machine Learning.

## 1. Dataset

Trong supervised learning:

$$\mathcal{D} = \{(\mathbf{x}^{(1)},y^{(1)}),\ldots,(\mathbf{x}^{(n)},y^{(n)})\}$$

- $n$: số samples
- $\mathbf{x}^{(i)}$: feature vector của sample thứ $i$
- $y^{(i)}$: target

---

## 2. Sample / Observation

Một sample là một đối tượng hoặc quan sát.

**Ví dụ một căn nhà:**

| Area | Bedrooms | Age | Price |
| :---: | :---: | :---: | :---: |
| 80 | 2 | 5 | 3.2 |

*Các từ gần nghĩa:* sample, observation, instance, example, data point.

---

## 3. Feature

Feature là biến đầu vào model dùng để dự đoán.

$$
\mathbf{x}^{(i)} =
\begin{bmatrix}
x_1^{(i)} \\
x_2^{(i)} \\
\vdots \\
x_d^{(i)}
\end{bmatrix}
$$


- $d$: số features
- $x_j^{(i)}$: feature thứ $j$ của sample $i$

---

## 4. Target / Label

### Regression
$$y \in \mathbb{R}$$

*Ví dụ:* giá nhà, nhiệt độ, doanh thu.

### Classification
$$y \in \{0,1,\ldots,K-1\}$$

*Ví dụ:* spam/not spam, fraud/normal.

> [!note] Note
> `label` thường dùng nhiều trong classification; `target` là cách gọi tổng quát hơn.

---

## 5. Feature Matrix ($X$)

Nếu có $n$ samples và $d$ features:

$$X \in \mathbb{R}^{n 	imes d}$$

- **hàng** = sample
- **cột** = feature

---

## 6. Target Vector ($\mathbf{y}$)

$$
\mathbf{y} =
\begin{bmatrix}
y^{(1)} \\
\vdots \\
y^{(n)}
\end{bmatrix}
$$


Trong scikit-learn thường:

```python
X.shape == (n_samples, n_features)
y.shape == (n_samples,)
```

---

## 7. Model và Prediction

Model học ánh xạ:

$$f: \mathcal{X} \rightarrow \mathcal{Y}$$

Sau training:

$$
\hat{y} = f_\theta(\mathbf{x})
$$



- $\theta$: parameters
- $\hat{y}$: prediction

---

## 8. Feature Types

* **Numerical:** continuous, discrete
* **Categorical:** nominal, ordinal
* **Datetime**
* **Text / Image / Audio:** thường cần representation trước khi vào model cổ điển

---

## 9. Lỗi thường gặp

* Dùng ID như một feature số có ý nghĩa thứ tự
* Dùng feature chỉ tồn tại sau prediction time
* Nhầm target-derived information thành feature

---

> [!summary] Tóm tắt
> **Supervised Learning cơ bản:**
> $$X 
\rightarrow 	{Model} \rightarrow \hat{y}$$