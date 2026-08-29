---
title: "Optimization Basics"
tags: [machine-learning, foundations, optimization]
status: complete
---
---
# Optimization Basics

## 1. Optimization trong Machine Learning

Quá trình huấn luyện mô hình (Training) thường được biểu diễn dưới dạng bài toán tối ưu hóa:

$$\theta^* = \arg\min_\theta J(\theta)$$

* **$\theta$:** Các tham số của mô hình (model parameters).
* **$J(\theta)$:** Hàm mục tiêu (objective function / loss function).

> [!note] Khái niệm cốt lõi
> Bài toán tối ưu hóa (Optimization) nhằm tìm ra bộ tham số $\theta^*$ sao cho giá trị của hàm mục tiêu $J(\theta)$ đạt mức nhỏ nhất.

---

## 2. Gradient (Vector độ dốc)

Vector gradient $\nabla_\theta J(\theta)$ là vector chứa các đạo hàm riêng theo từng tham số:

$$\nabla_\theta J(\theta) = \begin{bmatrix} \frac{\partial J}{\partial \theta_1} \\ \vdots \\ \frac{\partial J}{\partial \theta_p} \end{bmatrix}$$

* **Ý nghĩa địa hình:** Vector Gradient luôn chỉ theo **hướng tăng nhanh nhất** của hàm mục tiêu $J$.

---

## 3. Thuật toán Gradient Descent

Để làm giảm giá trị của hàm mục tiêu $J$, ta di chuyển tham số theo **hướng ngược lại** với Gradient:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)$$

* **$\eta$ (Learning Rate):** Tốc độ học / Bước nhảy.
* **$t$:** Vòng lặp thứ $t$ (iteration).

---
## 4. Ảnh hưởng của Learning Rate ($\eta$)

* **$\eta$ quá nhỏ:** Mô hình học rất chậm, tốn nhiều thời gian và vòng lặp tính toán để hội tụ.
* **$\eta$ quá lớn:** Bước nhảy quá xa có thể vượt qua điểm tối ưu (overshoot), làm giá trị hàm mục tiêu dao động mạnh hoặc thậm chí bị phân kỳ (không hội tụ).

---

## 5. Các biến thể Gradient Descent

### 5.1 Batch Gradient Descent
* Sử dụng **toàn bộ dataset** để tính toán Gradient cho mỗi lần cập nhật tham số.
* **Ưu điểm:** Cập nhật ổn định, hướng đi chính xác tới điểm tối ưu.
* **Nhược điểm:** Tốn kém tài nguyên tính toán và bộ nhớ khi tập dữ liệu lớn.

### 5.2 Stochastic Gradient Descent (SGD)
* Sử dụng **duy nhất 1 mẫu dữ liệu (sample)** để tính Gradient và cập nhật tham số ngay lập tức.
* **Ưu điểm:** Tốc độ cập nhật cực nhanh, dễ thoát khỏi các điểm cực trị địa phương.
* **Nhược điểm:** Quá trình hội tụ rất chập chùng và nhiều nhiễu (noisy).

### 5.3 Mini-batch Gradient Descent
* Sử dụng **một nhóm mẫu nhỏ (batch)** (ví dụ: 32, 64, 128) cho mỗi lần tính Gradient.
* **Đặc điểm:** Cân bằng giữa độ ổn định của Batch GD và tốc độ của SGD; là tiêu chuẩn phổ biến nhất trong Deep Learning hiện nay.

---

## 6. Bài toán Convex vs Non-convex
### Convex (Hàm lồi)
* Có tính chất toán học lý tưởng: **Local minimum cũng chính là Global minimum**.
### Non-convex (Hàm không lồi)
* Bề mặt tối ưu phức tạp, mô hình có thể bị kẹt tại các điểm:
  * **Local minima:** Điểm cực trị địa phương.
  * **Saddle points:** Điểm yên ngựa (gradient bằng 0 nhưng không phải cực trị).
  * **Flat regions:** Các vùng phẳng làm gradient gần bằng 0.

*Các mạng thần kinh nhân tạo (Neural Networks) hầu hết đều có bề mặt loss dạng Non-convex.*

---

## 7. Ma trận Hessian (Đạo hàm bậc hai)

Hessian $H$ là ma trận đạo hàm bậc hai mô tả độ cong (curvature) của hàm mục tiêu:

$$H = \nabla^2 J$$

* *Ứng dụng:* Thư viện **XGBoost** khai thác cả Gradient (đạo hàm bậc 1) và Hessian (đạo hàm bậc 2) của hàm loss để đưa ra quyết định xây dựng cây hiệu quả.

---

## 8. Không phải mô hình nào cũng dùng Gradient Descent

* **Linear / Logistic Regression:** Thường dùng các giải thuật dựa trên Gradient Descent (hoặc công thức đóng Normal Equation cho Linear Regression).
* **Neural Networks:** Dùng các biến thể nâng cao của Gradient Descent (Adam, RMSprop, SGD with Momentum).
* **Decision Tree:** **Không** dùng Gradient Descent; cây tìm điểm chia (split) bằng việc duyệt và tối ưu chỉ số Gini Impurity hoặc Information Gain.
* **XGBoost:** Dùng Gradient và Hessian của hàm loss theo cơ chế Boosting để tính điểm lá (leaf output) và Gain, chứ không dùng Gradient Descent để tối ưu trực tiếp ngưỡng chia (split thresholds).

---

## 9. Điều kiện dừng thuật toán Optimization

Quá trình tối ưu hóa thường dừng lại khi đạt một trong các điều kiện sau:
* Giá trị hàm mục tiêu $J(\theta)$ thay đổi không đáng kể giữa các vòng lặp.
* Độ lớn của Gradient $\|\nabla J\|$ tiến rất gần về 0.
* Đạt số lượng vòng lặp tối đa được cấu hình (`max_iterations`).
* Kích hoạt cơ chế **Early Stopping** (khi hiệu năng trên tập validation bắt đầu giảm).

---

> [!summary] Tóm tắt tư duy cốt lõi
> $$\text{Objective } J(\theta) \longrightarrow \text{Gradient } \nabla J \longrightarrow \text{Update } \theta \longrightarrow \text{Repeat}$$