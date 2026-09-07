---
title: "XGBoost — Lý thuyết trực quan và cách mô hình hoạt động"
aliases:
  - XGBoost
  - Extreme Gradient Boosting
tags:
  - machine-learning
  - supervised-learning
  - ensemble-learning
  - gradient-boosting
  - decision-tree
  - regression
  - classification
  - time-series
status: complete
source:
  - XGBoost_v3.pdf
---

# XGBoost — Lý thuyết trực quan và cách mô hình hoạt động

> [!NOTE] Mục tiêu của ghi chú
> Ghi chú này tập trung vào việc giúp bạn hiểu:
>
> - XGBoost đang học điều gì ở mỗi vòng lặp?
> - Tại sao mô hình phải xây nhiều cây nối tiếp nhau?
> - XGBoost Regression hoạt động như thế nào?
> - XGBoost Classification khác Regression ở đâu?
> - Residual, gradient, Hessian, Similarity Score và Gain có vai trò gì?
> - `lambda`, `gamma`, `learning_rate` và `min_child_weight` ảnh hưởng ra sao?
> - Khi đã dùng thư viện có sẵn, điều gì quan trọng nhất trong công việc thực tế?
>
> Trọng tâm không phải tự viết lại bộ tối ưu của XGBoost từ đầu, mà là hiểu đủ sâu để:
>
> - Chuẩn bị dữ liệu đúng
> - Chọn objective đúng
> - Tune hyperparameter có lý do
> - Phát hiện overfitting
> - Giải thích dự đoán
> - Đưa mô hình vào pipeline thực tế

---

# Phần I — Tổng quan về XGBoost

## 1. XGBoost là gì?

**XGBoost** là viết tắt của **Extreme Gradient Boosting**.

Đây là một thuật toán thuộc nhóm:

- **Supervised Learning**
- **Ensemble Learning**
- **Boosting**
- **Decision Tree–based Model**

XGBoost không xây một cây thật lớn để dự đoán ngay từ đầu.

Thay vào đó, mô hình xây nhiều cây nhỏ theo thứ tự:

```text
Dự đoán ban đầu
      ↓
Cây 1 học phần sai còn lại
      ↓
Cập nhật dự đoán
      ↓
Cây 2 học phần sai còn lại mới
      ↓
Cập nhật dự đoán
      ↓
...
      ↓
Tổng hợp tất cả cây
```

> [!NOTE] Ý tưởng cốt lõi
> Mỗi cây mới không dự đoán lại mục tiêu từ đầu.
>
> Nó học cách **sửa lỗi của tổng các cây đã có trước đó**.

---

## 2. [[Ensemble Learning|Boosting]] khác [[Ensemble Learning|Bagging]] như thế nào?

### Bagging

Ví dụ điển hình: **Random Forest**.

Các cây:

- Được xây tương đối độc lập
- Có thể train song song
- Cuối cùng vote hoặc lấy trung bình

```text
Tree 1 ─┐
Tree 2 ─┼──> Vote / Average
Tree 3 ─┘
```

Mục tiêu chính thường là giảm variance nhờ trung bình hóa nhiều cây khác nhau.

### Boosting

Trong boosting, các cây phụ thuộc tuần tự:

```text
Tree 1 → Tree 2 sửa Tree 1 → Tree 3 sửa tổng trước đó → ...
```

Cây sau cần biết mô hình hiện tại đang sai ở đâu.

> [!IMPORTANT]
> Trong XGBoost, thứ tự của cây có ý nghĩa.
>
> Bạn không thể đổi vị trí các cây một cách tùy ý như một tập các mô hình hoàn toàn độc lập.

---

## 3. Mô hình cộng dồn

Dự đoán sau cây thứ $t$:

$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(\mathbf{x}_i)
$$

Dự đoán cuối cùng:

$$
\hat{y}_i = \hat{y}_i^{(0)} + \eta \sum_{t=1}^{T} f_t(\mathbf{x}_i)
$$

Trong đó:
* $\hat{y}_i^{(0)}$: dự đoán ban đầu
* $f_t$: cây thứ $t$
* $\eta$: learning rate
* $T$: tổng số cây

> [!NOTE]
> Mỗi cây chỉ đóng góp một phần vào dự đoán cuối cùng.
>
> `learning_rate` kiểm soát mức độ ta tin vào cây mới.

---

## 4. Vì sao XGBoost cần Regularization?

Bài giảng bắt đầu bằng vấn đề **overfitting / high variance**.

Một mô hình có thể:

- Fit rất tốt dữ liệu train
- Nhưng dự đoán kém trên dữ liệu mới

XGBoost tối ưu:

$$
\text{Objective} = \text{Training Loss} + \text{Regularization}
$$

Dạng tổng quát:

$$
\mathcal{Obj}^{(t)} = \sum_{i=1}^{n} \mathcal{L}\left(y_i, \hat{y}_i^{(t)}\right) + \Omega(f_t)
$$

Regularization của một cây:

$$
\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T}w_j^2
$$

Trong đó:
* $T$: số lá của cây
* $w_j$: output của lá $j$
* $\gamma$: chi phí tạo thêm lá hoặc nhánh
* $\lambda$: regularization trên leaf output

> [!NOTE]
> XGBoost không chỉ hỏi:
>
> “Split này làm loss giảm bao nhiêu?”
>
> Mà còn hỏi:
>
> “Mức cải thiện có đủ lớn để xứng đáng với độ phức tạp tăng thêm không?”

![Pasted image 20260804110932](../../../../05_Assets/Images/Pasted%20image%2020260804110932.png)
![Pasted image 20260804110949](../../../../05_Assets/Images/Pasted%20image%2020260804110949.png)
![Pasted image 20260804111001](../../../../05_Assets/Images/Pasted%20image%2020260804111001.png)
![Pasted image 20260804111021](../../../../05_Assets/Images/Pasted%20image%2020260804111021.png)

---

# Phần II — XGBoost Regression

# 5. Tổng quan XGBoost Regression

Trong Regression, mục tiêu là dự đoán một giá trị liên tục.

Ví dụ trong bài giảng:

- Input: `Drug Weight`
- Output: `Drug Effectiveness`

## Quy trình:
**Step1:** Khởi tạo giá trị $f0$ dự đoán của model bằng cách lấy trung bình của $Y$.
**Step2:** Tính toán Similarity Score của root:$$\text{Similarity Score} = \frac{(\text{Sum of Residuals})^2}{\text{Number of Residuals} + \lambda}$$
* $\text{Sum of Residuals}$ là tổng của các giá trị trong $Y - f0$.
* $\text{Number of Residuals}$ là số lượng sample.
**Step3:** Có nhiều cách chọn điều kiện root, cơ bản nhất là lấy trung bình của 2 sample liên tiếp nhau. Sau đó tính Similarity Score cho các node trong nhánh trái và nhánh phải.
**Step4:** Tính Gain cho từng điều kiện của root đã chọn ở trên và chọn ra Gain có giá trị lớn nhất:
$$\text{Gain} = \text{Left Similarity Score} + \text{Right Similarity Score} - \text{Root Similarity Score}$$
**Step5:** Tuỳ vào điều kiện độ sâu của tree mà ta sẽ thực hiện chia nhánh bằng cách lặp lại Step2 đến Step4. Sau đó ta đi tìm output cho root theo điều kiện có gain lớn nhất:
$$\text{Output} = \frac{\text{Sum of Residuals}}{\text{Number of Residuals}}$$
* $\text{Sum of Residuals}$ là tổng của các giá trị trong $Y - f0$.
* $\text{Number of Residuals}$ là số lượng sample.
**Step6:** Dùng công thức sau $f0 + lr * \text{Output}$ để dự đoán kết quả cho toàn bộ training sample (thay thế cho $f0$) và tiếp tục thực hiện step 2 đến step 5 cho đến khi thoả mãn điều kiện dừng.

![Pasted image 20260804145414](../../../../05_Assets/Images/Pasted%20image%2020260804145414.png)

---

## 6. Bước 1 — Khởi tạo dự đoán ban đầu

Ta bắt đầu bằng một giá trị dự đoán chung:

$$
\hat{y}_i^{(0)} = c
$$

Trong ví dụ bài giảng:

$$
\hat{y}_i^{(0)} = 0.5
$$

Đây chỉ là điểm xuất phát. Các cây sau sẽ dần sửa prediction này.
![Pasted image 20260804145444](../../../../05_Assets/Images/Pasted%20image%2020260804145444.png)

---

### 7. Residual trong Regression

Với squared error:

$$
r_i = y_i - \hat{y}_i
$$

**Ý nghĩa:**
* $r_i > 0$: dự đoán đang thấp
* $r_i < 0$: dự đoán đang cao
* $r_i \approx 0$: dự đoán gần đúng

Trong Regression với squared error, có thể hiểu trực quan:
> Cây mới học các residual của ensemble hiện tại.

Tuy nhiên, cách tổng quát hơn là XGBoost học từ gradient và Hessian.
Residual rất khác nhau, triệt tiêu lẫn nhau, nên SC nhỏ
Residual giống nhau hoặc triệt tiêu lẫn nhau, nên SC khá lớn.

---

### 8. Gradient và Hessian trong Regression

Giả sử loss:

$$
\mathcal{L} = \frac{1}{2} (y_i - \hat{y}_i)^2
$$

Gradient:

$$
g_i = \frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \hat{y}_i - y_i
$$

Vì $r_i = y_i - \hat{y}_i$ nên:

$$
g_i = -r_i
$$

Hessian:

$$
h_i = 1
$$

Do đó:
* Gradient là âm residual ($g_i = -r_i$)
* Hessian bằng 1 cho mọi mẫu

> [!NOTE] Important
> * Với **squared error**, giải thích “cây học residual” là đúng về mặt trực giác.
> * Với **objective khác**, nên nghĩ theo gradient và Hessian.

---

### 9. Taylor bậc hai trong XGBoost

Tại vòng $t$:

$$
\mathcal{L}\left(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\right)
$$

được xấp xỉ:

$$
\mathcal{L}\left(y_i, \hat{y}_i^{(t-1)}\right) + g_i f_t(\mathbf{x}_i) + \frac{1}{2}h_i f_t(\mathbf{x}_i)^2
$$

Trong đó:

$$
g_i = \frac{\partial \mathcal{L}(y_i, \hat{y}_i)}{\partial \hat{y}_i}
$$

$$
h_i = \frac{\partial^2 \mathcal{L}(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}
$$

**Trực giác:**
* **Gradient:** cho biết prediction cần tăng hay giảm.
* **Hessian:** cho biết loss nhạy đến mức nào với thay đổi prediction.

---

### 10. Similarity Score trong Regression

Bài giảng dùng Similarity Score để đo một nhóm residual có phù hợp nằm chung trong một node hay không.

Với squared error:

$$
\text{Similarity Score} = \frac{\left( \sum r_i \right)^2}{m + \lambda}
$$

Trong đó:
* $m$: số residual trong node
* $r_i$: residual
* $\lambda$: L2 regularization

**Trực giác:**
Nếu residual:
* Cùng dấu
* Có xu hướng giống nhau
* Không triệt tiêu nhau

Thì $\left(\sum r_i\right)^2$ lớn, nên score lớn.

Nếu residual trái dấu và triệt tiêu nhau, score nhỏ.

---

### 11. Gain trong Regression

Khi thử một split:

$$
\text{Gain}_{\text{raw}} = \text{Score}_{\text{left}} + \text{Score}_{\text{right}} - \text{Score}_{\text{parent}}
$$

Dạng tổng quát có regularization:

$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

XGBoost thử các threshold và chọn split có Gain lớn nhất.

* Threshold 15 $\rightarrow$ Gain 120.33
* Threshold 22.5 $\rightarrow$ Gain 4.00
* Threshold 30 $\rightarrow$ Gain 56.33

$\Rightarrow$ **Chọn threshold 15**

**Summary:** Split tốt là split gom các mẫu có hướng sửa lỗi tương tự vào cùng một node.

---

### 12. Tiếp tục xây cây Regression

Sau khi chọn split đầu tiên:
* Một leaf có thể dừng
* Leaf còn lại tiếp tục thử split
* XGBoost lại tính Similarity Score và Gain

Quá trình tiếp tục cho đến khi:
* Đạt `max_depth`
* Không còn split có Gain đủ tốt
* Node không đạt điều kiện tối thiểu
* Hoặc cấu trúc bị pruning

---

### 13. Vai trò của $\gamma$ trong Regression

$\gamma$ là chi phí để tạo thêm một split.

$$
\text{Difference} = \text{Gain} - \gamma
$$

* Nếu $\text{Difference} > 0$: không xóa nhánh.
* Nếu $\text{Difference} \le 0$: xóa nhánh.

**Tác động:**
* **Khi $\gamma$ lớn:** Ít split hơn, cây đơn giản hơn, regularization mạnh hơn, có thể giảm overfitting.
* **Khi $\gamma$ nhỏ:** Dễ tạo split, cây phức tạp hơn, có thể fit train tốt hơn, nguy cơ overfitting cao hơn.

> [!NOTE] Important
> $\gamma = 0$ không có nghĩa là mọi split đều được giữ. Nếu raw Gain âm, split vẫn bị loại.
> Dùng cross validation để tìm $\gamma$ 

> [!TIP] Hình minh họa nên thêm (PDF trang 24–27 và 31–32)
> * Gain so với $\gamma$
> * Nhánh bị pruning
> * Trường hợp $\gamma = 0$ nhưng split vẫn bị loại

---

### 14. Vai trò của $\lambda$ trong Regression

Similarity Score:

$$
\text{Score} = \frac{G^2}{H + \lambda}
$$

Leaf output:

$$
w^* = -\frac{G}{H + \lambda}
$$

Với squared error:

$$
w^* = \frac{\sum r_i}{m + \lambda}
$$

**Khi tăng $\lambda$:**
* Score giảm
* Gain thường giảm
* Leaf output nhỏ hơn
* Model bớt nhạy với node nhỏ
* Overfitting có thể giảm

**Summary:**
* $\lambda$: regularize giá trị output của lá ($w$).
* $\gamma$: regularize cấu trúc cây ($T$).

---

### 15. Output tối ưu của một lá Regression

Đặt:

$$
G_j = \sum_{i \in I_j} g_i
$$

$$
H_j = \sum_{i \in I_j} h_i
$$

Leaf output tối ưu:

$$
w_j^* = -\frac{G_j}{H_j + \lambda}
$$

Với squared error:

$$
w_j^* = \frac{\sum_{i \in I_j} r_i}{|I_j| + \lambda}
$$

* Nếu $\lambda = 0$, output lá là **trung bình residual**.
* Nếu $\lambda > 0$, output bị **co lại gần 0**.

---

### 16. Cập nhật prediction Regression

Sau khi cây trả về leaf output:

$$
\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta f_t(\mathbf{x})
$$

**Ví dụ:**
* Base prediction: $0.5$
* Leaf output: $-10.5$
* Learning rate: $0.3$

$$
\hat{y}^{(1)} = 0.5 + 0.3(-10.5) = -2.65
$$

Sau đó cây tiếp theo học phần sai còn lại mới.

---

### 17. Learning Rate trong Regression

$$
\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta f_t(\mathbf{x})
$$

* **Learning rate nhỏ ($\eta$ nhỏ):** Mỗi cây sửa ít, cần nhiều cây hơn, học ổn định hơn. Thường kết hợp tốt với *early stopping*.
* **Learning rate lớn ($\eta$ lớn):** Mỗi cây sửa mạnh, cần ít cây hơn, dễ overfit hoặc cập nhật quá mức.

> [!NOTE] Important
> `learning_rate` ($\eta$) và `n_estimators` (số lượng cây $T$) phải được cân chỉnh cùng nhau.

---

## 18. Quantile Sketch trong Regression

Khi feature có rất nhiều giá trị khác nhau, thử mọi threshold có thể quá tốn kém.

Bài giảng nhắc đến:

> **Quantile Sketch Approximate Solution**

Ý tưởng:

- Chọn các candidate split đại diện
- Không cần thử toàn bộ giá trị
- Giảm chi phí tính toán
- Vẫn giữ khả năng tìm split tốt

---

## 19. Tóm tắt quy trình Regression

```text
Base prediction
      ↓
Tính residual / gradient / Hessian
      ↓
Tạo root node
      ↓
Thử threshold
      ↓
Tính Similarity Score
      ↓
Tính Gain
      ↓
Chọn split tốt nhất
      ↓
Regularize bằng λ và γ
      ↓
Tính leaf output
      ↓
Nhân learning rate
      ↓
Cập nhật prediction
      ↓
Xây cây tiếp theo
```

---

# Phần III — XGBoost Classification

# 20. Tổng quan XGBoost Classification

Với binary classification, mục tiêu là dự đoán xác suất thuộc lớp dương.

Ví dụ trong bài giảng:

- `Effective`
- `Not Effective`

## Quy trình:
**Step1:** Khởi tạo giá trị $f0$ dự đoán của model thông thường là $0.5$ (xác suất của dự đoán).
**Step2:** Tính toán Similarity Score của root:
$$\text{Similarity Score} = \frac{(\text{Sum of Residuals})^2}{\sum [\text{PreviousProbability} * (1 - \text{PreviousProbability})] + \lambda}$$
* $\text{Sum of Residuals}$ là tổng của các giá trị trong $Y$ (ở dạng xác suất) $- f0$.
* $\text{PreviousProbability}$ là xác suất của model trước dự đoán ví dụ ở đây là $0.5$.
**Step3:** Có nhiều cách chọn điều kiện root, cơ bản nhất là lấy trung bình của 2 sample liền kề. Sau đó tính Similarity Score cho các node trong nhánh trái và nhánh phải.
**Step4:** Tính Gain cho từng điều kiện của root đã chọn ở trên và chọn ra Gain có giá trị lớn nhất:$$\text{Gain} = \text{Left Similarity Score} + \text{Right Similarity Score} - \text{Root Similarity Score}$$
**Step5:** Tuỳ vào điều kiện độ sâu của tree mà ta sẽ thực hiện chia nhánh bằng cách lặp lại Step2 đến Step4. Sau đó ta đi tìm output cho root theo điều kiện có gain lớn nhất:
$$\text{Output} = \frac{\text{Sum of Residuals}}{\sum [\text{PreviousProbability} * (1 - \text{PreviousProbability})]}$$
* $\text{Sum of Residuals}$ là tổng của các giá trị trong $Y$ (ở dạng xác suất) $- f0$.
* $\text{PreviousProbability}$ là xác suất của model trước dự đoán ví dụ ở đây là $0.5$.
**Step6:** Dùng công thức bên dưới để dự đoán kết quả (Probability) cho toàn bộ training sample (thay thế cho $f0$) và tiếp tục thực hiện step 2 đến step 5 cho đến khi thoả mãn điều kiện dừng:
* Cần chọn nhánh phù hợp theo giá trị của $X$.
* Khi chọn được nhánh phù hợp ta tính Log prediction theo công thức:
  $$\text{LogPrediction} = \log\left(\frac{\text{PreviousProbability}}{1 - \text{PreviousProbability}}\right) + lr * \text{Output}$$
  * $\log$ là Natural logarithm.
* Xác suất dự đoán của các sample được tính theo công thức:
  $$\text{Probability} = \frac{e^{\text{LogPrediction}}}{1 + e^{\text{LogPrediction}}}$$
Điểm khác quan trọng:

> Trong Classification, cây không cộng trực tiếp vào probability.
>
> Cây cộng vào **raw score / log-odds**.

![Pasted image 20260804160952](../../../../05_Assets/Images/Pasted%20image%2020260804160952.png)

---

## 21. Probability, Odds và Log-odds

**Odds:**
$$\text{odds} = \frac{\hat{y}}{1-\hat{y}}$$

**Log-odds (Logit):**
$$\log(odds) = \log\left(\frac{\hat{y}}{1-\hat{y}}\right)$$

**Chuyển $\log(odds)$ về probability ($\hat{y}$):**
$$\hat{y} = \frac{1}{1+e^{-\log(odds)}}$$

> [!NOTE] Quy tắc Ensemble
> Trong **binary classification**, các mô hình ensemble sẽ cộng dồn trên raw score ($z$), không phải cộng dồn trên xác suất ($\hat{y}$).

---

## 22. Dự đoán ban đầu (Classification)

Nếu xác suất ban đầu là $\hat{y}_0 = 0.5$, thì raw score ban đầu $z_0$ được tính bằng:
$$\log(odds)_0 = \log\left(\frac{0.5}{1-0.5}\right) = 0$$

*Lưu ý: Nếu tỷ lệ lớp dương (positive class) khác 0.5, base raw score $z_0$ sẽ khác 0.*

> [!NOTE] Hình minh họa cần thêm (Trang 51–53 trong PDF)
> - [ ] Chuyển probability sang odds
> - [ ] Chuyển odds sang log-odds

---

## 23. Loss, Gradient và Hessian Classification

**Binary Log Loss (Cross-Entropy Loss):**
$$\mathcal{L}(y, \hat{y}) = - \left[ y\log(\hat{y}) + (1-y)\log(1-\hat{y}) \right]$$

Trong đó dự đoán xác suất $\hat{y}$ được tính qua hàm Sigmoid của $\log(odds)$:
$$\hat{y} = \sigma(\log(odds)) = \frac{1}{1+e^{-\log(odds)}}$$

* **Gradient** (Đạo hàm bậc 1 của Loss theo $z$):
  $$g_i = \frac{\partial \mathcal{L}}{\partial \log(odds)_i} = \hat{y}_i - y_i$$

* **Residual (Độ lệch trực quan):**
  $$r_i = y_i - \hat{y}_i = -g_i$$

* **Hessian** (Đạo hàm bậc 2 của Loss theo $z$):
  $$h_i = \frac{\partial^2 \mathcal{L}}{\partial z_i^2} = \hat{y}_i(1-\hat{y}_i)$$

### So sánh với Regression
* **Regression (Squared Error):** $h_i = 1$
* **Classification (Logistic):** $h_i = \hat{y}_i(1-\hat{y}_i)$

---

## 24. Similarity Score Classification

**Dạng tổng quát:**
$$\text{Similarity Score} = \frac{G^2}{H+\lambda}$$

Trong đó:
$$G = \sum_i g_i \quad \text{và} \quad H = \sum_i h_i$$

**Viết chi tiết theo Residual:**
$$\text{Similarity Score} = \frac{\left( \sum_i (y_i - \hat{y}_i) \right)^2}{\sum_i \hat{y}_i(1-\hat{y}_i) + \lambda}$$

> [!IMPORTANT] Điểm quan trọng
> Mẫu số của Classification **không đơn giản là số lượng mẫu**.
> Nó chính là **tổng Hessian**: 
> $$\sum_i \hat{y}_i(1-\hat{y}_i)$$


---

## 25. Gain Classification

Công thức Gain giữ nguyên cấu trúc tổng quát:
$$
\text{Gain} = \text{Score}_{\text{left}} + \text{Score}_{\text{right}} - \text{Score}_{\text{parent}}
$$

Điểm khác biệt nằm ở giá trị của $g_i$ và $h_i$:
* $g_i = \hat{y}_i - y_i$
* $h_i = \hat{y}_i(1-\hat{y}_i)$

*XGBoost sẽ thử nghiệm nhiều threshold khác nhau và chọn threshold có **Gain** lớn nhất.*

---

## 26. Cover trong Classification

Cách ước lượng số lương Residual ở mỗi lá => XGBoost Cover
Trong Classification, **Cover** được tính bằng:
$$\text{Cover} = \sum_i \hat{y}_i(1-\hat{y}_i)$$

Đây chính là tổng Hessian trong node ($H$).
Giá trị nhỏ nhất của Cover là 1, nếu Cover bé hơn 1 thì  xóa nhánh đó

**Ví dụ:**
Nếu $\hat{y}_i = 0.5$, ta có $\hat{y}_i(1-\hat{y}_i) = 0.25$.
Với một node chứa $4$ mẫu:
$$\text{Cover} = 4 \times 0.25 = 1$$
---

## 27. `min_child_weight` trong Classification

`min_child_weight` đặt ngưỡng tối thiểu cho tổng Hessian của node con.

### Giá trị lớn

- Node con cần Cover lớn hơn
- Khó tạo leaf nhỏ
- Cây bảo thủ hơn
- Có thể giảm overfitting

### Giá trị nhỏ

- Dễ tạo leaf nhỏ
- Model linh hoạt hơn
- Có thể overfit nhóm dữ liệu nhỏ

> [!IMPORTANT]
> `min_child_weight` không nhất thiết là số mẫu tối thiểu.
>
> Nó liên quan đến **tổng Hessian**.

> [!NOTE] Hình minh họa nên thêm
> Lấy hình từ **PDF trang 46–50**:
>
> - Cover của leaf
> - Minimum XGBoost Cover
> - Node được giữ hoặc bị xóa

---

## 28. Leaf output Classification

**Leaf output tối ưu:**
$$output = -\frac{G}{H+\lambda}$$

**Viết theo residual:**
$$output = \frac{\sum_i(y_i-\hat{y}_i)}{\sum_i \hat{y}_i(1-\hat{y}_i)+\lambda}$$

Output này nằm trong không gian:
$$\text{log-odds}$$
không phải probability.

---

## 29. Cập nhật log-odds

Sau khi có leaf output:
$$\text{LogPrediction} = \log\left(\frac{\text{PreviousProbability}}{1 - \text{PreviousProbability}}\right) + \eta * \text{Output}$$

Sau đó đổi về probability:
$$p_{\text{new}} = \frac{1}{1+e^{-\log(odds)_i{\text{new}}}}$$

> [!WARNING] Warning
> Không cộng leaf output trực tiếp vào probability.

---

## 30. Cây tiếp theo trong Classification

Sau khi cập nhật probability:

1. **Tính residual mới:**
   $$y_i-\hat{y}_i$$

2. **Tính Hessian mới:**
   $$\hat{y}_i(1-\hat{y}_i)$$

3. **Xây cây mới**
4. **Tính Gain**
5. **Tính leaf output mới**
6. **Cập nhật log-odds**

Quá trình tiếp tục cho đến khi dừng.

---

## 31. Tóm tắt quy trình Classification

```text
Base probability
      ↓
Chuyển sang log-odds
      ↓
Tính gradient và Hessian
      ↓
Tạo cây bằng Score và Gain
      ↓
Tính leaf output
      ↓
Nhân learning rate
      ↓
Cộng vào log-odds
      ↓
Sigmoid → probability mới
      ↓
Xây cây tiếp theo
```

---

## 32. So sánh Regression và Classification

| Thành phần           | Regression           | Binary Classification      |
| -------------------- | -------------------- | -------------------------- |
| Objective thường gặp | Squared error        | Binary log loss            |
| Prediction nội bộ    | Giá trị liên tục     | Log-odds                   |
| Gradient             | \(\hat y-y\)         | \(p-y\)                    |
| Hessian              | \(1\)                | \(p(1-p)\)                 |
| Leaf denominator     | Số mẫu + \(\lambda\) | Tổng Hessian + \(\lambda\) |
| Output cuối          | Giá trị số           | Sigmoid → probability      |
| Residual trực quan   | \(y-\hat y\)         | \(y-p\)                    |

---

# Phần IV — Cơ chế chung và tối ưu của XGBoost

## 33. Công thức tổng quát

**Mô hình:**
$$\hat{y}_i = \sum_{k=1}^{K} f_k(\mathbf{x}_i)$$

**Hàm mục tiêu (Objective function):**
$$\mathcal{Obj} = \sum_{i=1}^{n} \mathcal{L}(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

**Tại vòng thứ $t$:**
$$\mathcal{Obj}^{(t)} = \sum_{i=1}^{n} \mathcal{L}\left(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)$$

**Xấp xỉ Taylor bậc hai:**
$$\mathcal{Obj}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t(\mathbf{x}_i)^2 \right] + \Omega(f_t) + \text{constant}$$

**Với một cây có $T$ lá:**
$$f_t(\mathbf{x}) = w_{q(\mathbf{x})}$$

**Objective rút gọn:**
$$\widetilde{\mathcal{Obj}} = \sum_{j=1}^{T} \left[ G_j w_j + \frac{1}{2}(H_j + \lambda)w_j^2 \right] + \gamma T$$

**Leaf output tối ưu:**
$$w_j^* = -\frac{G_j}{H_j + \lambda}$$
---

## 34. Một vòng huấn luyện hoàn chỉnh

```text
1. Có prediction hiện tại
        ↓
2. Tính gradient và Hessian
        ↓
3. Bắt đầu từ root node
        ↓
4. Thử feature và threshold
        ↓
5. Tính Score parent / left / right
        ↓
6. Tính Gain
        ↓
7. Chọn split tốt nhất
        ↓
8. Kiểm tra γ, max_depth, min_child_weight
        ↓
9. Tính output tối ưu cho các lá
        ↓
10. Nhân learning rate
        ↓
11. Cập nhật prediction
        ↓
12. Xây cây tiếp theo
```

---

## 35. Khi nào dừng huấn luyện?

- Đạt số cây tối đa
- Validation metric không còn cải thiện
- Early stopping kích hoạt
- Không còn split có Gain hợp lệ
- Cây đạt `max_depth`
- Node không đạt `min_child_weight`

> [!IMPORTANT]
> Early stopping giúp chọn số boosting rounds dựa trên validation data thay vì đoán trước.

---

# Phần V — Hyperparameter

## 36. Các hyperparameter quan trọng

### `n_estimators`

Số cây boosting.

- Quá ít → underfitting
- Quá nhiều → có thể overfitting

### `learning_rate`

Mức đóng góp của mỗi cây.

- Nhỏ → cần nhiều cây
- Lớn → học nhanh nhưng rủi ro cao hơn

### `max_depth`

Độ sâu tối đa của mỗi cây.

- Lớn → học interaction phức tạp
- Nhỏ → cây đơn giản hơn

### `min_child_weight`

Tổng Hessian tối thiểu ở node con.

### `gamma`

Gain tối thiểu để giữ split.

### `reg_lambda`

L2 regularization trên leaf output.

### `reg_alpha`

L1 regularization trên leaf output.

### `subsample`

Tỷ lệ hàng dùng cho mỗi cây.

### `colsample_bytree`

Tỷ lệ feature dùng cho mỗi cây.

---

## 37. Nhóm hyperparameter theo chức năng

### Kiểm soát tốc độ học

- `learning_rate`
- `n_estimators`

### Kiểm soát độ phức tạp cây

- `max_depth`
- `min_child_weight`
- `gamma`

### Regularization leaf output

- `reg_lambda`
- `reg_alpha`

### Sampling

- `subsample`
- `colsample_bytree`

---

## 38. Cách suy nghĩ khi tune

### Underfitting

Dấu hiệu:

- Train kém
- Validation cũng kém

Có thể thử:

- Tăng số cây
- Tăng `max_depth`
- Giảm `min_child_weight`
- Giảm `gamma`
- Giảm regularization

### Overfitting

Dấu hiệu:

- Train rất tốt
- Validation kém hơn đáng kể

Có thể thử:

- Giảm `max_depth`
- Tăng `min_child_weight`
- Tăng `gamma`
- Tăng `reg_lambda` / `reg_alpha`
- Giảm `subsample`
- Giảm `colsample_bytree`
- Giảm learning rate
- Dùng early stopping

---
