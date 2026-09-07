---
tags: [optimization, machine-learning, deep-learning, gradient-descent, calculus, algorithms]
aliases: [Gradient Descent, Thuật toán hạ độ dốc, GD, SGD, Adam]
date: 2026-09-04
---

# Gradient Descent (Thuật toán Hạ Độ Dốc)

## 1. Đặt Vấn đề và Trực giác Hình học

### 1.1. Động lực
Trong học máy, hầu hết các bài toán đều quy về việc cực tiểu hóa một hàm mất mát (Cost / Loss Function) $f(\theta)$ đối với vector tham số $\theta \in \mathbb{R}^D$:

$$\theta^* = \arg\min_{\theta} f(\theta)$$

Điều kiện cần để một điểm là cực trị địa phương là đạo hàm bậc nhất triệt tiêu:

$$\nabla_\theta f(\theta) = 0$$

Tuy nhiên, trong đa số bài toán thực tế, phương trình trên không thể giải trực tiếp bằng phương pháp giải tích (closed-form solution) do hàm số phi tuyến hoặc có số lượng biến quá lớn. Do đó, phương pháp giải tích số lặp qua từng bước như **Gradient Descent** là giải pháp bắt buộc.

### 1.2. Trực giác Hình học
* Đạo hàm bậc nhất biểu diễn độ dốc và chiều biến thiên của hàm số tại một điểm.
* Vector gradient $\nabla f(\theta)$ luôn chỉ theo hướng **hàm số tăng nhanh nhất**.
* Để tìm giá trị nhỏ nhất, ta phải di chuyển ngược chiều với vector gradient (hướng giảm nhanh nhất của hàm số).

> [!NOTE] Cần ảnh minh họa: Trực quan hóa đường dốc 2D và mặt cong thung lũng 3D, với vector gradient trỏ lên đỉnh đồi và bước nhảy của thuật toán di chuyển ngược lại về phía đáy thung lũng.

---

## 2. Công thức Cập nhật và Các Yếu tố Ảnh hưởng

### 2.1. Công thức Cập nhật Tổng quát
Tại mỗi bước lặp thứ $t$, vector tham số $\theta$ được cập nhật theo quy tắc:

$$\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)$$

Trong đó:
* $\theta_t$: Vector tham số tại thời điểm $t$.
* $\eta > 0$: **Learning Rate (Tốc độ học)** - quy định độ lớn của mỗi bước nhảy.
* $\nabla f(\theta_t)$: Vector gradient của hàm số tại $\theta_t$.

### 2.2. Điều kiện Dừng Thuật toán (Stopping Criteria)
Thuật toán sẽ dừng khi thỏa mãn một trong các điều kiện sau:
1. **Chuẩn của Gradient đủ nhỏ:** Khi độ dốc gần như bằng phẳng:
   $$\|\nabla f(\theta_t)\|_2 < \varepsilon$$
   *(với $\varepsilon$ là một ngưỡng sai số rất nhỏ, ví dụ $10^{-3}$ hoặc $10^{-5}$)*.
2. **Số vòng lặp tối đa:** Đạt ngưỡng số bước lặp định trước (`max_iter`).
3. **Mức độ thay đổi hàm mục tiêu không đáng kể:** $|f(\theta_{t+1}) - f(\theta_t)| < \delta$.

### 2.3. Tác động của Learning Rate ($\eta$)

| Giá trị $\eta$ | Hiện tượng | Hệ quả |
| :--- | :--- | :--- |
| **Quá nhỏ (Too Small)** | Di chuyển cực kỳ chậm chạp | Mất rất nhiều vòng lặp để hội tụ; dễ bị nghẽn ở vùng bằng phẳng. |
| **Quá lớn (Too High)** | Bước nhảy vượt quá đáy cực tiểu (Overshooting) | Dao động mạnh quanh điểm tối ưu, thậm chí phân kỳ (Divergence) và bùng nổ loss. |
| **Hợp lý (Optimal)** | Giảm đều qua từng bước | Hội tụ trơn tru và nhanh chóng về điểm tối ưu. |

> [!NOTE] Cần ảnh minh họa: Đồ thị so sánh 3 trường hợp của Learning Rate: Learning rate quá nhỏ (bước rùa bò), tối ưu (hội tụ mượt mà) và quá lớn (nhảy vọt văng ra ngoài hàm mục tiêu).

### 2.4. Vấn đề Điểm Khởi tạo ($\theta_0$)
* **Với hàm lồi (Convex):** Điểm cực tiểu địa phương cũng chính là điểm cực tiểu toàn cục. Mọi điểm khởi tạo $\theta_0$ đều dẫn đến cùng một nghiệm tối ưu.
* **Với hàm phi lồi (Non-convex):** Các điểm khởi tạo khác nhau sẽ dẫn đến các điểm cực tiểu địa phương khác nhau (Local Minima).

> [!NOTE] Cần ảnh minh họa: Đồ thị hàm phi lồi 1D với hai điểm bắt đầu khác nhau dẫn về hai hố cực tiểu địa phương khác nhau.

---

## 3. Ba Biến thể Chính của Gradient Descent

### 3.1. Batch Gradient Descent (BGD)
* **Cách hoạt động:** Sử dụng toàn bộ $N$ điểm dữ liệu trong dataset để tính đạo hàm trung bình cho một lần cập nhật tham số:
  $$\nabla f(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla f_i(\theta)$$
* **Ưu điểm:** Hướng di chuyển rất chính xác, đường suy giảm loss mượt mà.
* **Nhược điểm:** Tốc độ tính toán chậm và tốn bộ nhớ RAM khi $N$ lên tới hàng triệu mẫu. Không phù hợp với dữ liệu cập nhật theo luồng (Online Learning).

### 3.2. Stochastic Gradient Descent (SGD)
* **Cách hoạt động:** Tại mỗi bước lặp, chọn ngẫu nhiên đúng **1 mẫu dữ liệu** $(x_i, y_i)$ để tính gradient xấp xỉ và cập nhật tham số ngay lập tức:
  $$\theta_{t+1} = \theta_t - \eta \nabla f_i(\theta_t)$$
* **Ưu điểm:** Chi phí tính toán cực rẻ, cập nhật tham số ngay lập tức. Tính ngẫu nhiên (noise) trong từng bước có thể giúp mô hình "văng" ra khỏi các điểm cực tiểu địa phương nông.
* **Nhược điểm:** Quỹ đạo di chuyển zíc zắc dữ dội và không ổn định quanh điểm tối ưu nếu không áp dụng kỹ thuật giảm dần learning rate (Learning Rate Decay).

### 3.3. Mini-batch Gradient Descent
* **Cách hoạt động:** Chia tập dữ liệu thành các gói nhỏ (mini-batch) có kích thước $B$ (thường là $32, 64, 128, 256$):
  $$\theta_{t+1} = \theta_t - \eta \frac{1}{B} \sum_{i \in \text{Batch}} \nabla f_i(\theta_t)$$
* **Ưu thế:** Kết hợp sự ổn định của BGD và tốc độ của SGD, tận dụng tối đa năng lực xử lý ma trận song song trên GPU. Đây là tiêu chuẩn vàng trong huấn luyện Deep Learning hiện đại.

> [!NOTE] Cần ảnh minh họa: Đồ thị so sánh quỹ đạo di chuyển trên đường đồng mức (Contour Plot) giữa 3 phương pháp: BGD (đường đi thẳng tắp mượt mà), SGD (đường đi zíc zắc hỗn loạn) và Mini-batch (quỹ đạo cân bằng ổn định).

---

## 4. Các Thách thức Lớn đối với GD Cổ điển

> [!WARNING] Những cạm bẫy trong không gian tối ưu hóa
> 1. **Saddle Points (Điểm Yên ngựa):** Những điểm có gradient bằng 0 ($\nabla f(\theta) = 0$), nhưng lại là cực đại theo chiều này và cực tiểu theo chiều khác. GD cổ điển sẽ bị "đóng băng" tại đây do không còn lực kéo.
> 2. **Ravines (Hẻm vực hẹp):** Bề mặt có độ dốc theo chiều này lớn gấp nhiều lần chiều kia. GD cổ điển sẽ bị dội qua lại giữa hai vách đá thay vì đi dọc theo lòng đáy hẻm.

> [!NOTE] Cần ảnh minh họa: Hình ảnh bề mặt Điểm Yên ngựa (Saddle Point) và Hẻm vực dốc hẹp (Ravines/Pathological Curvatures) minh họa việc gradient descent bị dao động ngang.

---

## 5. Các Thuật toán Tối ưu Hóa Hiện đại (Modern Optimizers)

### 5.1. Gradient Descent with Momentum
Mô phỏng hòn bi kim loại có quán tính lăn xuống dốc. Ngoài lực kéo tức thời của gradient tại bước hiện tại, vật thể còn duy trì vận tốc tích lũy từ các bước trước đó.

* **Công thức toán học:**
  $$v_t = \gamma v_{t-1} + \eta \nabla f(\theta_t)$$
  $$\theta_{t+1} = \theta_t - v_t$$
  *(Trong đó $\gamma \approx 0.9$ là hệ số suy giảm quán tính)*.
* **Tác dụng:** Triệt tiêu các dao động zíc zắc đổi hướng liên tục ở hẻm vực, đồng thời tích lũy đủ vận tốc để vượt qua các gờ bằng phẳng hoặc điểm yên ngựa.

### 5.2. Nesterov Accelerated Gradient (NAG)
Là phiên bản "nhìn trước" (look-ahead) của Momentum. Thay vì tính gradient tại vị trí hiện tại $\theta_t$, NAG tính gradient tại vị trí dự đoán mà quán tính sẽ đưa cá thể tới:

* **Công thức toán học:**
  $$v_t = \gamma v_{t-1} + \eta \nabla f(\theta_t - \gamma v_{t-1})$$
  $$\theta_{t+1} = \theta_t - v_t$$
* **Tác dụng:** Hoạt động như một cơ chế "hãm phanh thông minh", giảm bớt quán tính khi sắp lao quá đà qua đáy thung lũng.

> [!NOTE] Cần ảnh minh họa: Sơ đồ hình học phân tích vector so sánh giữa bước nhảy Momentum cổ điển và bước nhảy nhìn trước của Nesterov (NAG).

### 5.3. AdaGrad (Adaptive Gradient Algorithm)
Tự động thích nghi tốc độ học riêng cho từng tham số dựa vào lịch sử bình phương gradient của chính tham số đó.

* **Công thức toán học:**
  $$G_t = G_{t-1} + (\nabla f(\theta_t))^2$$
  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla f(\theta_t)$$
* **Nhược điểm cốt tử:** Do $G_t$ là tổng cộng dồn dương liên tục, mẫu số ngày càng lớn khiến learning rate hiệu dụng tiệm cận 0 quá sớm, làm thuật toán dừng lại trước khi chạm đáy tối ưu.

### 5.4. RMSprop (Root Mean Square Propagation)
Khắc phục nhược điểm giảm tốc quá sớm của AdaGrad bằng cách thay thế tổng cộng dồn bằng **trung bình trượt phân rã theo hàm mũ (Exponential Moving Average)**:

* **Công thức toán học:**
  $$v_t = \beta v_{t-1} + (1 - \beta) (\nabla f(\theta_t))^2$$
  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \odot \nabla f(\theta_t)$$
  *(Thường chọn $\beta \approx 0.9$)*.

### 5.5. Adam (Adaptive Moment Estimation)
Thuật toán tối ưu hóa tiêu chuẩn và được ưa chuộng nhất hiện nay trong học sâu. Adam kết hợp ưu điểm của cả **Momentum** (ước lượng kỳ vọng gradient - moment bậc 1) và **RMSprop** (ước lượng phương sai gradient - moment bậc 2), đồng thời bổ sung bước hiệu chỉnh độ lệch (bias correction).

* **Công thức toán học đầy đủ:**
  1. *Moment bậc 1:*
     $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla f(\theta_t)$$
  2. *Moment bậc 2:*
     $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla f(\theta_t))^2$$
  3. *Hiệu chỉnh bias cho các bước đầu tiên:*
     $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  4. *Cập nhật tham số:*
     $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
  *(Tham số mặc định chuẩn: $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$)*.

---

## 6. Code Minh họa: Cài đặt Gradient Descent cơ bản với NumPy

```python
import numpy as np

# 1. Định nghĩa hàm mục tiêu f(x) = x^2 + 5*sin(x) và đạo hàm f'(x)
def grad(x):
    return 2 * x + 5 * np.cos(x)

def cost(x):
    return x**2 + 5 * np.sin(x)

# 2. Thuật toán Gradient Descent 1D
def gradient_descent_1d(eta, x_init, max_iter=1000, tol=1e-3):
    x_history = [x_init]
    for i in range(max_iter):
        x_new = x_history[-1] - eta * grad(x_history[-1])
        if abs(grad(x_new)) < tol:
            break
        x_history.append(x_new)
    return x_new, i + 1, x_history

# Chạy thử nghiệm với các điểm khởi tạo khác nhau
x_init = 5.0
eta = 0.1
x_opt, iters, _ = gradient_descent_1d(eta, x_init)

print(f"Nghiệm tối ưu tìm được: x = {x_opt:.4f}")
print(f"Giá trị hàm mục tiêu: f(x) = {cost(x_opt):.4f}")
print(f"Số bước lặp hội tụ: {iters}")
```
