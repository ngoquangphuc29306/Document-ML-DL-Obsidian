---
tags:
  - ensemble-learning
  - classification
  - regression
  - supervised-learning
created: 2026-07-24
---

# <font color = "red">AdaBoost</font>

## 1. What is AdaBoost ?
**AdaBoost (Adaptive Boosting)** là thuật toán tiêu biểu và cổ điển nhất thuộc họ [[Ensemble Learning|Boosting]] trong Học máy (Machine Learning).
### Ý tưởng:
- **Học từ sai lầm (Adaptive):** Thuật toán hoạt động theo chuỗi nối tiếp qua các vòng lặp. Sau mỗi vòng, AdaBoost điều chỉnh **trọng số mẫu (`sample weight`)** — tức mức độ ưu tiên của từng điểm dữ liệu trong huấn luyện.
* **Tăng trọng số dữ liệu khó:** Các điểm dữ liệu bị phân loại sai hoặc có sai số lớn sẽ được **tăng trọng số**, buộc mô hình yếu (**weak learner**) ở vòng tiếp theo phải chú trọng nhiều hơn vào khu vực khó này.
* **Kết hợp có trọng số:** Tất cả các weak learners được gộp lại bằng phép cộng có trọng số (với hệ số ảnh hưởng `amount of say` khác nhau), tạo nên một mô hình mạnh (**strong learner**) có độ chính xác vượt trội.
![Pasted image 20260724204915](../../../../05_Assets/Images/Pasted%20image%2020260724204915.png)

### Decision Stump
Trong thực tế, weak learner thường được chọn là **Decision Stump** (Gốc cây quyết định / Cây quyết định độ sâu 1):
* **Đặc điểm:** Chỉ sử dụng **một đặc trưng (feature)** duy nhất và **một ngưỡng chia (threshold)** để phân tách dữ liệu thành hai nhóm.
* **Vai trò:** Dù là một mô hình cực kỳ đơn giản (chỉ nhỉnh hơn đoán ngẫu nhiên một chút), khi kết hợp hàng trăm/hàng ngàn decision stumps qua từng vòng huấn luyện, tổ hợp này sẽ giải quyết được các bài toán phi tuyến phức tạp.

### Bảng ký hiệu toán học

|                  Ký hiệu                  | Ý nghĩa                                                                                                                |
| :---------------------------------------: | :--------------------------------------------------------------------------------------------------------------------- |
|                    $N$                    | Tổng số mẫu trong tập dữ liệu huấn luyện.                                                                              |
|                    $M$                    | Số vòng lặp (số weak learners) của mô hình.                                                                            |
|  $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$   | Tập dữ liệu huấn luyện; $x_i$ là đặc trưng, $y_i$ là nhãn.                                                             |
|                   $y_i$                   | Nhãn: M1 $y_i \in \{-1, +1\}$; R2 $y_i \in \mathbb{R}$.                                                                |
|   $y_i' = 2 \cdot \text{target}_i - 1$    | Mã hoá nhị phân từ $\text{target} \in \{0, 1\}$ sang $\{-1, +1\}$.                                                     |
|                 $h_m(x)$                  | Weak learner (decision stump) ở vòng $m$.                                                                              |
|                $w_i^{(m)}$                | Trọng số của mẫu $i$ tại vòng $m$, $\sum_i w_i^{(m)} = 1$.                                                             |
|              $\varepsilon_m$              | Lỗi có trọng số: M1 $\sum_{i=1}^N w_i^{(m)} \mathbb{I}\{h_m(x_i) \neq y_i\}$; R2 $\sum_i w_i^{(m)} e_i$.               |
|                $\alpha_m$                 | Trọng số phiếu: M1 $\frac{1}{2} \ln \frac{1 - \varepsilon_m}{\varepsilon_m}$; R2 $\ln \frac{1}{\beta_m}$.              |
|                 $F_m(x)$                  | Mô hình cộng dồn: $F_m(x) = \sum_{t=1}^m \alpha_t h_t(x)$.                                                             |
|               $\hat{y}(x)$                | Dự đoán cuối: (M1) $= \text{sign}(F_M(x))$; (R2) $= \text{wmed}\left(\{h_m(x)\}_{m=1}^M; \{\alpha_m\}_{m=1}^M\right)$. |
|                   $Z_m$                   | Hằng số chuẩn hoá khi cập nhật trọng số: $Z_m = \sum_i w_i^{(m)} \Psi(\cdot)$.                                         |
|                   $e_i$                   | Sai số đã chuẩn hoá (R2): $e_i = \frac{\|y_i - h_m(x_i)\|}{\max_j \|y_j - h_m(x_j)\|} \in [0, 1]$.                     |
|                 $\beta_m$                 | Tỉ lệ lỗi (R2): $\beta_m = \frac{\varepsilon_m}{1 - \varepsilon_m} \in (0, 1)$.                                        |
|             $\mathcal{T}, t$              | Tập ngưỡng ứng viên và một ngưỡng stump trên một đặc trưng liên tục.                                                   |
|               $R_\ell, R_r$               | Hai miền tách bởi ngưỡng $t$: $R_\ell = \{x < t\}$, $R_r = \{x \ge t\}$.                                               |
| $\tilde{y}_\ell^{(w)}, \tilde{y}_r^{(w)}$ | Trung bình (có trọng số) của mục tiêu trong từng lá.                                                                   |

---

## 2. AdaBoost trong Classification (Phân Loại Nhị Phân - M1)
### 2.1. Phát biểu bài toán
Cho tập dữ liệu $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ với $x_i \in \mathbb{R}^n, y_i \in \{-1, +1\}$. Mục tiêu là học một tổ hợp tuyến tính các weak learners $h_m: \mathbb{R}^n\rightarrow\{-1, +1\}$ theo dạng cộng dồn:
$$F_m(x) = \sum_{t=1}^m \alpha_t h_t(x)$$
![Pasted image 20260724210606](../../../../05_Assets/Images/Pasted%20image%2020260724210606.png)

### 2.2. Thuật toán chi tiết
#### **Bước 1: Khởi tạo**
- Gán trọng số ban đầu đồng đều cho tất cả các mẫu:
$$w_i^{(1)} = \frac{1}{N}, \qquad F_0(x) \equiv 0, \qquad y_i \in \{-1, +1\}.$$
- Với phân phối trọng số công bằng này, ta huấn luyện weak learner đầu tiên để có đánh giá khởi điểm.
> *Lưu ý:* Nếu dữ liệu gốc hoặc weak learner trả về nhãn $\{0, 1\}$, quy đổi về $\{-1, +1\}$ bằng công thức $y \leftarrow 2y - 1$ và $h_m \leftarrow 2h_m - 1$.

---
#### **Bước 2: Lặp theo từng vòng $m = 1, 2, \dots, M$**
##### **(a) Huấn luyện Weak Classifier $h_m$**
Huấn luyện $h_m(x)$ trên tập dữ liệu với phân phối trọng số $w^{(m)}$. Decision Stump chọn thuộc tính và ngưỡng sao cho giảm thiểu lỗi phân loại có trọng số (hoặc tối ưu Gini có trọng số).
##### **(b) Tính Weighted Error ($\varepsilon_m$)**
- **Weighted error** $\varepsilon_m$ đo sai số của $h_m$ theo phân phối trọng số $w_i^{(m)}$: mẫu có trọng số lớn ảnh hưởng nhiều hơn. Đối với Binary Classification, đó là tổng trọng số các mẫu bị phân loại sai.
$$\varepsilon_m = \sum_{i=1}^N w_i^{(m)} \mathbb{I}\{h_m(x_i) \neq y_i\}.$$
Trong đó:
- $\varepsilon_m$ : weighted error của week learner ở vòng m
- $\mathbb{I}\{\cdot\}$ là hàm chỉ thị (bằng 1 nếu sai, 0 nếu đúng).
- $w_i^{(m)} >= 0$ : sample weight thứ i tại vòng m
- $h_m(x)$: week learner ở vòng m; dự đoán $h_m(x_i)$.
##### **(c) Xác định Amount of Say ($\alpha_m$)**

- Hệ số $\alpha_m$ thể hiện "tiếng nói" hay mức độ đóng góp của $h_m$ vào mô hình tổng hợp $F_m(x)$:
- $\alpha_m$ càng lớn thì "lá phiếu" của $h_m$ càng nặng, từ đó tác động mạnh hơn cả lên giá trị $F_m(x)$ lẫn bước cập nhật **sample weight**
$$\alpha_m=\frac{1}{2} \ln \frac{1 - \varepsilon_m}{\varepsilon_m}$$
<div align="center"> <img src="Pasted image 20260724213612.png" width="350"> </div>
###### **Chú ý**: 
- Nếu $\varepsilon_m=0$ có thể dừng sớm hoặc chặn $\alpha_m$ 
- Nếu $\varepsilon_m>=0.5$ và có thể lật dấu, đặt $h_m \leftarrow -h_m$ rồi cập nhật $\varepsilon_m \leftarrow 1-\varepsilon_m$ trước khi tính $\alpha_m$.
##### **(d) Cập nhật mô hình cộng dồn**
$$F_m(x) = F_{m-1}(x) + \alpha_m h_m(x)$$
- Bước này cập nhật điểm số cộng dồn $F_m$. Trọng số vòng sau được cập nhật trực tiếp từ $h_m$ theo câu thức bên dưới
##### **(e) Cập nhật trọng số mẫu ($w^{(m+1)}$) cho vòng $m+1$**

$$w_i^{(m+1)} = \frac{w_i^{(m)} \exp\left( -\alpha_m y_i h_m(x_i) \right)}{Z_m}, \qquad Z_m = \sum_{j=1}^N w_j^{(m)} \exp\left( -\alpha_m y_j h_m(x_j) \right).$$

Với $\varepsilon_m = \sum_i w_i^{(m)} \mathbf{1}\{h_m(x_i) \neq y_i\}$ và $\alpha_m = \frac{1}{2} \ln \frac{1 - \varepsilon_m}{\varepsilon_m}$, ta có

$$Z_m = (1 - \varepsilon_m) e^{-\alpha_m} + \varepsilon_m e^{+\alpha_m} = 2\sqrt{\varepsilon_m (1 - \varepsilon_m)}.$$

Tương đương:

$$w_i^{(m+1)} \propto \begin{cases} w_i^{(m)} e^{-\alpha_m}, & \text{nếu } h_m(x_i) = y_i, \\ w_i^{(m)} e^{+\alpha_m}, & \text{nếu } h_m(x_i) \neq y_i. \end{cases}$$

Trong đó:
* $w_i^{(m+1)}$: trọng số của mẫu $i$ sau khi cập nhật ở vòng $m$.
* $w_i^{(m)}$: trọng số của mẫu $i$ tại đầu vòng $m$.
* $\alpha_m$: hệ số *amount of say* của weak learner ở vòng $m$.
* $\exp(\cdot)$: hàm mũ cơ số $e$.
* $Z_m = \sum_{j=1}^N w_j^{(m)} \exp\left( -\alpha_m y_j h_m(x_j) \right)$: hằng số chuẩn hoá.

Sau khi chuẩn hoá để $\sum_i w_i^{(m+1)} = 1$, quy tắc cập nhật có ý nghĩa như sau: các mẫu được phân loại **đúng** ở vòng $m$ sẽ bị **giảm** trọng số, còn các mẫu bị phân loại **sai** sẽ được **tăng** trọng số. Nhờ đó, ở vòng kế tiếp weak classifier buộc phải chú ý nhiều hơn đến những điểm "khó".
![Pasted image 20260724215543](../../../../05_Assets/Images/Pasted%20image%2020260724215543.png)

---
#### Bước 3: Dự đoán cuối
Sau $M$ vòng, dự đoán cuối cùng $\hat{y}(x)$ là **kết quả bỏ phiếu có trọng số** của các weak learner: mỗi mô hình $h_m(x) \in \{-1, +1\}$ "bỏ phiếu" với trọng số $\alpha_m$, tạo thành
$$\hat{y}(x) = \text{sign}\left(F_M(x)\right).$$
Trong đó:$$\text{sign}\left(F_M(x)\right) = \begin{cases} +1 & \text{nếu } F_M(x) \ge 0, \\ -1 & \text{nếu } F_M(x) < 0. \end{cases}$$
Ở đây $\alpha_m$ càng lớn thì lá phiếu của nó càng nặng; dấu của $F_M(x)$ quyết định lớp thắng, còn $|F_M(x)|$ phản ánh độ tự tin của cuộc "bỏ phiếu".

---
## 2.3 Mối liên hệ giữa AdaBoost Classification và Exponential Loss
AdaBoost không chỉ có thể được hiểu theo trực giác là “tăng trọng số cho các mẫu bị phân loại sai”, mà còn có thể được xem như một thuật toán xây dựng mô hình cộng dồn nhằm giảm **exponential loss**.
Với mô hình cộng dồn:
$$
F_m(x)
=
\sum_{t=1}^{m}\alpha_t h_t(x),
$$
hàm mất mát mũ trên tập huấn luyện được định nghĩa là:
$$
\mathcal{L}(F)
=
\sum_{i=1}^{N}
\exp\left(-y_iF(x_i)\right).
$$
Tại vòng lặp thứ $m$, mô hình được cập nhật theo:
$$
F_m(x)
=
F_{m-1}(x)
+
\alpha_m h_m(x).
$$
Thay vào exponential loss:
$$
\mathcal{L}(F_m)
=
\sum_{i=1}^{N}
\exp
\left[
-y_i
\left(
F_{m-1}(x_i)
+
\alpha_m h_m(x_i)
\right)
\right].
$$
Tách hàm mũ thành tích:
$$
\mathcal{L}(F_m)
=
\sum_{i=1}^{N}
\exp\left(-y_iF_{m-1}(x_i)\right)
\exp\left(-\alpha_m y_i h_m(x_i)\right).
$$
Đặt trọng số của mẫu $i$ tại vòng $m$ tỉ lệ với:
$$
w_i^{(m)}
\propto
\exp\left(-y_iF_{m-1}(x_i)\right),
$$
ta thu được bài toán tối ưu tại vòng $m$:
$$
\mathcal{L}(F_m)
\propto
\sum_{i=1}^{N}
w_i^{(m)}
\exp\left(-\alpha_m y_i h_m(x_i)\right).
$$
Do đó, quy tắc cập nhật trọng số mẫu có dạng:
$$
w_i^{(m+1)}
\propto
w_i^{(m)}
\exp\left(-\alpha_m y_i h_m(x_i)\right).
$$
Vì $y_i,h_m(x_i)\in\{-1,+1\}$ nên:
$$
y_i h_m(x_i)
=
\begin{cases}
+1, & \text{nếu } h_m(x_i)=y_i,\\
-1, & \text{nếu } h_m(x_i)\neq y_i.
\end{cases}
$$
Suy ra:
$$
w_i^{(m+1)}
\propto
\begin{cases}
w_i^{(m)}e^{-\alpha_m},
& \text{nếu mẫu được phân loại đúng},\\[4pt]
w_i^{(m)}e^{+\alpha_m},
& \text{nếu mẫu bị phân loại sai}.
\end{cases}
$$
Như vậy, quy tắc tăng trọng số cho mẫu sai và giảm trọng số cho mẫu đúng xuất hiện trực tiếp từ việc tối ưu exponential loss.

---

### Suy ra công thức $\alpha_m$
Sau khi chọn được weak classifier $h_m$, ta cần tìm hệ số $\alpha_m$ sao cho hàm mất mát tại vòng $m$ là nhỏ nhất.
Xét hàm:
$$
J(\alpha)
=
\sum_{i=1}^{N}
w_i^{(m)}
\exp\left(-\alpha y_i h_m(x_i)\right).
$$
Gọi weighted error của $h_m$ là:
$$
\varepsilon_m
=
\sum_{i=1}^{N}
w_i^{(m)}
\mathbb{I}
\left\{
h_m(x_i)\neq y_i
\right\}.
$$
Vì các trọng số đã được chuẩn hóa:
$$
\sum_{i=1}^{N}w_i^{(m)}=1,
$$
nên tổng trọng số của các mẫu được phân loại đúng là:
$$
1-\varepsilon_m.
$$
Đối với các mẫu được phân loại đúng:
$$
y_i h_m(x_i)=+1,
$$
còn đối với các mẫu bị phân loại sai:
$$
y_i h_m(x_i)=-1.
$$
Do đó, ta có thể viết:
$$
J(\alpha)
=
(1-\varepsilon_m)e^{-\alpha}
+
\varepsilon_m e^{\alpha}.
$$
Lấy đạo hàm theo $\alpha$:

$$
J'(\alpha)
=
-(1-\varepsilon_m)e^{-\alpha}
+
\varepsilon_m e^{\alpha}.
$$

Để tìm giá trị cực tiểu, cho:
$$
J'(\alpha)=0.
$$
Khi đó:
$$
-(1-\varepsilon_m)e^{-\alpha}
+
\varepsilon_m e^{\alpha}
=
0.
$$
Suy ra:
$$
\varepsilon_m e^{\alpha}
=
(1-\varepsilon_m)e^{-\alpha}.
$$
Nhân hai vế với $e^{\alpha}$:
$$
\varepsilon_m e^{2\alpha}
=
1-\varepsilon_m.
$$
Do đó:
$$
e^{2\alpha}
=
\frac{1-\varepsilon_m}{\varepsilon_m}.
$$
Lấy logarit tự nhiên hai vế:

$$
2\alpha
=
\ln
\left(
\frac{1-\varepsilon_m}{\varepsilon_m}
\right).
$$

Vì vậy:

$$
\boxed{
\alpha_m
=
\frac{1}{2}
\ln
\left(
\frac{1-\varepsilon_m}{\varepsilon_m}
\right)
}.
$$

Công thức này cho thấy:

- Nếu $\varepsilon_m<\frac{1}{2}$ thì $\alpha_m>0$.
- Nếu $\varepsilon_m=\frac{1}{2}$ thì $\alpha_m=0$.
- Nếu $\varepsilon_m>\frac{1}{2}$ thì $\alpha_m<0$.
- Khi $\varepsilon_m$ càng nhỏ, $\alpha_m$ càng lớn và weak classifier có ảnh hưởng càng mạnh trong mô hình tổng hợp.
### 2.3 Python Code:

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Tạo dữ liệu giả lập (Binary Classification)
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=8, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Khởi tạo và huấn luyện AdaBoost Classifier
model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
model.fit(X_train, y_train)

# 3. Dự đoán và đánh giá
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
```

## 3. AdaBoost.R2 trong Regression

- Cho tập dữ liệu huấn luyện $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N, x_i \in \mathbb{R}^n$.
- Với bài toán Regression, ta học một dãy các weak regressors $h_m : \mathbb{R}^n \to \mathbb{R}$ cùng trọng số $\alpha_m \ge 0$. 
- Khác với AdaBoost Classification, AdaBoost Regression **không** dùng $F_M(x) = \sum_m \alpha_m h_m(x)$ làm dự đoán cuối cùng mà sử dụng **weighted median** (tạm dịch: trung vị có trọng số) của các giá trị $h_m(x)$ theo trọng số $\alpha_m$:

$$\widehat{y}(x) = \text{wmed}\left( \{h_m(x)\}_{m=1}^M ; \{\alpha_m\}_{m=1}^M \right).$$

Ở đây, wmed là giá trị $t$ nhỏ nhất sao cho:

$$\sum_{m: h_m(x) \le t} \alpha_m \ge \frac{1}{2} \sum_{m=1}^M \alpha_m.$$
### Bước 1: Khởi tạo

Gán trọng số đều cho các mẫu:
$$w_i^{(1)} = \frac{1}{N}, \quad i = 1, \dots, N.$$
Với phân phối trọng số công bằng này, ta huấn luyện weak learner đầu tiên để lấy đánh giá khởi điểm.
### Bước 2: Lặp theo vòng $m = 1, \dots, M$
Ở mỗi vòng, thuật toán huấn luyện một weak regressor theo phân phối trọng số hiện tại, chuẩn hóa lỗi của từng mẫu, tính weighted error, xác định trọng số đóng góp của weak regressor, lưu weak regressor vào ensemble, rồi cập nhật sample weight cho vòng kế tiếp.
#### Dự đoán tại lá
Nếu weak regressor là cây hồi quy sử dụng tiêu chí weighted MSE và nhận trực tiếp sample weight, giá trị dự đoán tối ưu tại mỗi vùng lá $R$ là weighted mean của các giá trị mục tiêu trong vùng:
$$
\widetilde{y}_{R}^{(w)}
=
\frac{
\sum_{i:x_i\in R} w_i^{(m)}y_i
}{
\sum_{i:x_i\in R} w_i^{(m)}
}.
$$
Nếu thuật toán được triển khai bằng weighted resampling thay vì truyền trực tiếp sample weight, cây được huấn luyện trên tập dữ liệu đã lấy mẫu lại và giá trị tại lá được tính theo tập dữ liệu lấy mẫu đó.
##### (a) **Huấn luyện weak learner**
Huấn luyện weak regressor
$$
h_m:\mathbb{R}^n\to\mathbb{R}
$$
theo phân phối trọng số hiện tại $w^{(m)}$.
Tùy cách triển khai, có thể:
- truyền trực tiếp $w_i^{(m)}$ dưới dạng `sample_weight` cho weak regressor; hoặc
- lấy mẫu lại dữ liệu theo xác suất $w_i^{(m)}$, rồi huấn luyện weak regressor trên tập dữ liệu được lấy mẫu.
##### (b) **Chuẩn hoá lỗi từng mẫu với linear loss:**
Đặt sai số tuyệt đối của mẫu $i$ tại vòng $m$ là:
$$
r_i^{(m)}
=
\left|y_i-h_m(x_i)\right|.
$$
Gọi sai số lớn nhất tại vòng $m$ là:
$$
R_m
=
\max_{1\le j\le N}r_j^{(m)}
=
\max_{1\le j\le N}
\left|y_j-h_m(x_j)\right|.
$$

Nếu $R_m>0$, lỗi chuẩn hóa của mẫu $i$ là:
$$
e_i^{(m)}
=
\frac{r_i^{(m)}}{R_m}
=
\frac{
\left|y_i-h_m(x_i)\right|
}{
\max_{1\le j\le N}
\left|y_j-h_m(x_j)\right|
}
\in[0,1].
$$
Trong đó:
* $e_i$: lỗi đã chuẩn hoá của mẫu $i$ (thuộc $[0, 1]$).
* $\max_{1 \le j \le N} |y_j - h_m(x_j)|$: mẫu số để chuẩn hoá sai số về thang $[0, 1]$.
* $w_i^{(m)}$: trọng số của mẫu $i$ tại đầu vòng $m$ (chuẩn hoá để $\sum_i w_i^{(m)} = 1$).
Nếu $R_m=0$, weak regressor dự đoán đúng toàn bộ các mẫu huấn luyện. Khi đó đặt
$$
e_i^{(m)}=0,\qquad \forall i,
$$
lưu $h_m$ vào ensemble và dừng boosting sớm.
##### (c) **Tính weighted error**
Weighted error của weak regressor $h_m$ là trung bình có trọng số của các lỗi chuẩn hóa:

$$
\varepsilon_m
=
\sum_{i=1}^{N}
w_i^{(m)}e_i^{(m)}
\in[0,1].
$$
Trong đó:
- $\varepsilon_m$: lỗi có trọng số của weak regressor tại vòng $m$;
- $e_i^{(m)}$: lỗi chuẩn hóa của mẫu $i$ tại vòng $m$;
- $w_i^{(m)}\ge0$: trọng số của mẫu $i$ tại vòng $m$;
- $\sum_{i=1}^{N}w_i^{(m)}=1$.
Để weak regressor được chấp nhận, cần:
$$
\varepsilon_m<\frac12.
$$
Nếu $\varepsilon_m\ge\frac12$, weak regressor hiện tại không đủ tốt và không được thêm vào ensemble.
##### (d) **Xác định amount of say**
Xét ba trường hợp:
- Nếu

$$
\varepsilon_m=0,
$$
weak regressor dự đoán hoàn hảo tập huấn luyện. Lưu $h_m$ vào ensemble và dừng boosting sớm.
- Nếu
$$
\varepsilon_m\ge\frac12,
$$
weak regressor không đủ tốt. Không thêm $h_m$ vào ensemble và dừng vòng boosting.
- Nếu
$$
0<\varepsilon_m<\frac12,
$$
tính:
$$
\beta_m
=
\frac{\varepsilon_m}{1-\varepsilon_m}
\in(0,1),
$$
và trọng số đóng góp của weak regressor:
$$
\alpha_m
=
\ln\left(\frac{1}{\beta_m}\right)
=
\ln\left(
\frac{1-\varepsilon_m}{\varepsilon_m}
\right)
>0.
$$
Trong đó:
- $\beta_m$: hệ số được sử dụng để cập nhật sample weight;
- $\alpha_m$: trọng số đóng góp của weak regressor $h_m$ khi tính weighted median;
- $\ln(\cdot)$: logarit tự nhiên.
Khi $\varepsilon_m$ càng nhỏ thì $\beta_m$ càng nhỏ và $\alpha_m$ càng lớn. Vì vậy, weak regressor có weighted error nhỏ sẽ có ảnh hưởng lớn hơn đến dự đoán cuối cùng.
##### (e) **Lưu weak learner vào ensemble**

Sau khi tính được trọng số $\alpha_m$, ta lưu cặp weak regressor và trọng số tương ứng vào danh sách mô hình:

$$
\mathcal{H}_m
=
\bigl(
(h_1,\alpha_1),
(h_2,\alpha_2),
\dots,
(h_m,\alpha_m)
\bigr).
$$

Hoặc có thể biểu diễn thao tác thêm mô hình mới vào cuối danh sách:

$$
\mathcal{H}_m
=
\mathcal{H}_{m-1}
\mathbin{\|}
(h_m,\alpha_m),
$$

trong đó:
- $\mathcal{H}_m$: danh sách có thứ tự gồm các weak regressors và trọng số của chúng sau $m$ vòng.
- $\mathcal{H}_{m-1}$: danh sách các weak regressors đã được lưu trước vòng $m$.
- $h_m$: weak regressor được huấn luyện tại vòng $m$.
- $\alpha_m$: trọng số đóng góp của $h_m$ trong bước dự đoán cuối.
- $\mathbin{\|}$: ký hiệu thao tác thêm phần tử $(h_m,\alpha_m)$ vào cuối danh sách.
Khác với AdaBoost Classification, AdaBoost.R2 không cần sử dụng tổng cộng dồn
$$
F_m(x)=\sum_{t=1}^{m}\alpha_t h_t(x)
$$

để tạo dự đoán cuối cùng. Thay vào đó, thuật toán lưu toàn bộ các cặp $(h_m,\alpha_m)$ và sử dụng **weighted median** của các dự đoán từ weak regressors:

$$
\widehat{y}(x)=\operatorname{wmed}\left(\{h_m(x)\}_{m=1}^{M};\{\alpha_m\}_{m=1}^{M}\right).
$$
(f) **Cập nhật sample weight cho vòng $m+1$:**
$$w_i^{(m+1)} = \frac{w_i^{(m)} \beta_m^{1 - e_i}}{Z_m}, \quad Z_m = \sum_{j=1}^N w_j^{(m)} \beta_m^{1 - e_j}, \quad \text{sao cho } \sum_i w_i^{(m+1)} = 1.$$

Với mẫu **dễ** ($e_i \approx 0$), hệ số nhân $\beta_m^{1 - e_i} \approx \beta_m < 1$ làm trọng số bị giảm. Ngược lại, với mẫu **khó** ($e_i \approx 1$), hệ số nhân xấp xỉ $1$ nên trọng số hầu như được giữ nguyên (tương đối tăng so với mẫu dễ). Nhờ vậy, vòng kế tiếp tập trung hơn vào các điểm/vùng còn khó.
### Bước 3: Dự đoán cuối
Sau $M$ vòng, với một mẫu đầu vào $x$, mỗi weak regressor tạo ra một giá trị dự đoán:
$$
h_1(x), h_2(x), \dots, h_M(x).
$$
AdaBoost.R2 không lấy trung bình có trọng số của các dự đoán này. Thay vào đó, dự đoán cuối cùng là **weighted median** (trung vị có trọng số) theo các trọng số $\alpha_m$:
$$
\widehat{y}(x)
=
\operatorname{wmed}
\left(
\{h_m(x)\}_{m=1}^{M};
\{\alpha_m\}_{m=1}^{M}
\right).
$$
Để xác định weighted median, trước hết sắp xếp các giá trị dự đoán theo thứ tự tăng dần:
$$
h_{(1)}(x)
\le
h_{(2)}(x)
\le
\dots
\le
h_{(M)}(x),
$$
trong đó $\alpha_{(1)}, \alpha_{(2)}, \dots, \alpha_{(M)}$ là các trọng số tương ứng sau khi sắp xếp.
Chọn chỉ số nhỏ nhất $k$ sao cho:
$$
\sum_{j=1}^{k}\alpha_{(j)}
\ge
\frac{1}{2}
\sum_{j=1}^{M}\alpha_{(j)}.
$$
Khi đó, dự đoán cuối cùng là:
$$
\widehat{y}(x)
=
h_{(k)}(x).
$$
Tương đương, weighted median có thể được viết dưới dạng:

$$
\widehat{y}(x)
=
\inf
\left\{
y \in \mathbb{R}
:
\sum_{m:h_m(x)\le y}\alpha_m
\ge
\frac{1}{2}
\sum_{m=1}^{M}\alpha_m
\right\}.
$$

Trong đó:
- $h_{(j)}(x)$: giá trị dự đoán đứng ở vị trí thứ $j$ sau khi sắp xếp tăng dần.
- $\alpha_{(j)}$: trọng số tương ứng với weak regressor tạo ra $h_{(j)}(x)$.
- $k$: vị trí đầu tiên mà tổng trọng số tích lũy đạt ít nhất một nửa tổng trọng số.
- Weak regressor có $\alpha_m$ lớn sẽ đóng góp nhiều hơn vào tổng trọng số tích lũy và do đó có ảnh hưởng lớn hơn đến vị trí của weighted median.
Việc sử dụng weighted median giúp AdaBoost.R2 ít bị ảnh hưởng bởi các dự đoán quá lớn hoặc quá nhỏ hơn so với việc sử dụng weighted mean.

### Ưu điểm của AdaBoost

* **Đơn giản và dễ triển khai:** Thuật toán có cấu trúc trực quan, ít tham số cần tinh chỉnh so với các mô hình phức tạp khác.
* **Ít bị overfitting:** Nhờ cơ chế kết hợp nhiều mô hình yếu (weak learners) và giảm trọng số của các cây khi tăng số lượng, AdaBoost kiểm soát hiện tượng quá khớp tương đối tốt trên các tập dữ liệu tiêu chuẩn.
* **Tăng độ chính xác cao:** Có khả năng biến các mô hình dự đoán yếu (như cây quyết định nông - decision stumps) thành một mô hình phân loại mạnh mẽ.
* **Tự động tập trung vào các mẫu khó:** Thuật toán tự động tăng trọng số cho các mẫu dự đoán sai ở bước trước, giúp các mô hình sau tập trung sửa lỗi hiệu quả.
* **Thích ứng linh hoạt:** Có thể kết hợp được với nhiều thuật toán học máy cơ sở khác nhau, không chỉ riêng cây quyết định.

---

### Nhược điểm của AdaBoost

* **Nhạy cảm với nhiễu và outlier:** Vì thuật toán liên tục tăng trọng số cho các mẫu bị đoán sai, các điểm dữ liệu nhiễu hoặc ngoại lệ (outliers) có thể chiếm trọng số quá lớn, làm lệch hướng học của mô hình.
* **Tốc độ huấn luyện chậm hơn khi dữ liệu lớn:** Do quá trình huấn luyện là tuần tự (mô hình sau phụ thuộc vào kết quả mô hình trước), AdaBoost khó thực hiện tính toán song song trên quy mô lớn.
* **Hiệu năng giảm trên dữ liệu mất cân bằng:** Nếu tập dữ liệu bị lệch lớp nghiêm trọng, thuật toán có thể quá tập trung vào lớp yếu số lượng ít mà bỏ qua tổng thể.
* **Phụ thuộc vào chất lượng mô hình cơ sở:** Nếu các mô hình yếu quá kém (độ chính xác dưới 50%) hoặc quá phức tạp (dễ bị overfit ngay từ đầu), AdaBoost sẽ không đạt được hiệu quả tối ưu.

![Pasted image 20260724223310](../../../../05_Assets/Images/Pasted%20image%2020260724223310.png)

![Pasted image 20260724235307](../../../../05_Assets/Images/Pasted%20image%2020260724235307.png)