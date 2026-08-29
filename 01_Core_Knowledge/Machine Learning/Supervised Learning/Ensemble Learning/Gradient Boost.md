---
tags:
  - ensemble-learning
  - machine-learning
  - classification
  - regression
created: 2026-07-25
---

# Gradient Boosting

## 1. Gradient Boosting là gì?

**Gradient Boosting** là một thuật toán thuộc nhóm [[Ensemble Learning|Boosting]], trong đó nhiều weak learners được huấn luyện **tuần tự** để tạo thành một mô hình mạnh.

Trong thực tế, weak learner thường là một **Regression Tree có độ sâu nhỏ**.
### Ý tưởng chính
Mỗi cây mới không dự đoán lại giá trị mục tiêu từ đầu. Thay vào đó, nó học cách sửa phần lỗi mà mô hình hiện tại vẫn còn dự đoán sai.
Quy trình có thể nhớ ngắn gọn như sau:
$$
\boxed{
\text{Dự đoán}
\rightarrow
\text{Tính phần sai}
\rightarrow
\text{Xây cây học phần sai}
\rightarrow
\text{Cập nhật dự đoán}
\rightarrow
\text{Lặp lại}
}
$$
Mô hình cuối cùng là tổng của:
- dự đoán ban đầu;
- phần điều chỉnh của cây thứ nhất;
- phần điều chỉnh của cây thứ hai;
- và các cây tiếp theo.
$$
F_M(x)
=
F_0(x)
+
\eta h_1(x)
+
\eta h_2(x)
+\dots+
\eta h_M(x).
$$

Trong đó:

- $F_0(x)$: dự đoán ban đầu;
- $F_m(x)$: dự đoán của toàn bộ mô hình sau vòng $m$;
- $h_m(x)$: cây thứ $m$, dùng để sửa lỗi của mô hình trước;
- $\eta$: learning rate;
- $M$: tổng số cây.

> [!important]
> Gradient Boosting không huấn luyện các cây độc lập.  
> Cây sau phụ thuộc vào kết quả của tất cả các cây trước đó.

---

## 2. Trực giác đơn giản

Giả sử cần dự đoán cân nặng của một người:

$$
y=88.
$$

Mô hình ban đầu dự đoán:

$$
F_0(x)=71.2.
$$

Phần mô hình còn dự đoán thiếu là:

$$
r_1
=
88-71.2
=
16.8.
$$

Cây thứ nhất sẽ cố gắng học giá trị điều chỉnh gần với $16.8$.

Nếu learning rate là:

$$
\eta=0.1,
$$

thì mô hình không cộng toàn bộ $16.8$, mà chỉ cộng:

$$
0.1\times16.8=1.68.
$$

Dự đoán mới là:

$$
F_1(x)
=
71.2+1.68
=
72.88.
$$

Sau đó cây thứ hai tiếp tục học phần sai còn lại:

$$
88-72.88=15.12.
$$

Như vậy, mỗi cây giúp mô hình tiến gần hơn đến giá trị thật.

---

## 3. Tập dữ liệu ví dụ

Ví dụ trong tài liệu sử dụng dữ liệu sau:

| Height | Favorite Color | Gender | Weight |
|---:|:---:|:---:|---:|
| 1.6 | Blue | Male | 88 |
| 1.6 | Green | Female | 76 |
| 1.5 | Blue | Female | 56 |
| 1.8 | Red | Male | 73 |
| 1.5 | Green | Male | 77 |
| 1.4 | Blue | Female | 57 |

Trong đó:

- `Height`, `Favorite Color`, `Gender` là các đặc trưng đầu vào;
- `Weight` là giá trị cần dự đoán;
- đây là một bài toán Regression.

Ta ký hiệu tập dữ liệu:

$$
\mathcal D
=
\{(x_i,y_i)\}_{i=1}^{N}.
$$

Trong đó:

- $x_i$: các đặc trưng của người thứ $i$;
- $y_i$: cân nặng thật;
- $N=6$: số lượng mẫu.

---

## 4. Loss function

Gradient Boosting cần một hàm mất mát để đo mức độ sai của mô hình.

Trong ví dụ Regression, ta sử dụng **Squared Error**:

$$
L(y_i,F(x_i))
=
\frac{1}{2}
\left(
y_i-F(x_i)
\right)^2.
$$

Trong đó:

- $y_i$: giá trị thật;
- $F(x_i)$: giá trị mô hình dự đoán;
- loss càng nhỏ thì dự đoán càng tốt.

Hệ số $\frac12$ được thêm vào để việc lấy đạo hàm thuận tiện hơn, nhưng không làm thay đổi vị trí cực tiểu.

---

## 5. Residual và pseudo-residual

### 5.1. Residual

Residual là chênh lệch giữa giá trị thật và giá trị dự đoán:

$$
r_i
=
y_i-\widehat y_i.
$$

Ví dụ:

$$
y_i=88,
\qquad
\widehat y_i=71.2.
$$

Khi đó:

$$
r_i
=
88-71.2
=
16.8.
$$

Ý nghĩa:

- $r_i>0$: mô hình dự đoán thấp hơn giá trị thật;
- $r_i<0$: mô hình dự đoán cao hơn giá trị thật;
- $r_i\approx0$: mô hình dự đoán khá chính xác.

### 5.2. Pseudo-residual

Trong Gradient Boosting tổng quát, cây mới học **negative gradient** của loss:

$$
r_{im}
=
-
\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]_{F=F_{m-1}}.
$$
``
$r_{im}$ được gọi là **pseudo-residual** của mẫu $i$ tại vòng $m$.

Với Squared Error:

$$
L(y_i,F(x_i))
=
\frac12
\left(y_i-F(x_i)\right)^2,
$$

ta có:

$$
\frac{\partial L}
{\partial F(x_i)}
=
F(x_i)-y_i.
$$

Do đó:

$$
r_{im}
=
-
\left(F_{m-1}(x_i)-y_i\right)
=
y_i-F_{m-1}(x_i).
$$

Vì vậy, với Squared Error:

$$
\boxed{
\text{Pseudo-residual}
=
\text{Residual thông thường}
}
$$

> [!note]
> Với các loss khác, pseudo-residual không nhất thiết bằng $y_i-F(x_i)$.  
> Vì vậy, Gradient Boosting tổng quát hơn việc chỉ học residual.

---

# 6. Thuật toán Gradient Boosting Regression

## Bước 1: Khởi tạo mô hình

Ta bắt đầu bằng một mô hình rất đơn giản, chỉ dự đoán một giá trị hằng số cho tất cả các mẫu:

$$
F_0(x)
=
\arg\min_{\gamma}
\sum_{i=1}^{N}
L(y_i,\gamma).
$$

Với Squared Error, giá trị tối ưu là trung bình của tất cả target:

$$
F_0(x)
=
\bar y
=
\frac{1}{N}
\sum_{i=1}^{N}y_i.
$$

Với dữ liệu ví dụ:

$$
\bar y
=
\frac{
88+76+56+73+77+57
}{6}.
$$

$$
\bar y
=
\frac{427}{6}
=
71.1667
\approx71.2.
$$

Do đó:

$$
\boxed{
F_0(x)=71.2
}
$$

Mô hình ban đầu dự đoán cân nặng của mọi người đều bằng khoảng $71.2$ kg.

---

## Bước 2: Tính residual lần thứ nhất

Residual của mỗi mẫu là:

$$
r_{i1}
=
y_i-F_0(x_i).
$$

Với:

$$
F_0(x_i)=71.2,
$$

ta có bảng residual gần đúng:

| Weight thật $y_i$ | Dự đoán $F_0(x_i)$ | Residual $r_{i1}$ |
|---:|---:|---:|
| 88 | 71.2 | 16.8 |
| 76 | 71.2 | 4.8 |
| 56 | 71.2 | -15.2 |
| 73 | 71.2 | 1.8 |
| 77 | 71.2 | 5.8 |
| 57 | 71.2 | -14.2 |

Ví dụ:

$$
r_{11}
=
88-71.2
=
16.8.
$$

$$
r_{31}
=
56-71.2
=
-15.2.
$$

Các residual này trở thành target mới để huấn luyện cây thứ nhất.

Tập dữ liệu mà cây thứ nhất học có dạng:

$$
\{
(x_i,r_{i1})
\}_{i=1}^{N}.
$$

> [!important]
> Cây không còn dự đoán trực tiếp `Weight`.  
> Cây dự đoán phần sai còn lại của mô hình hiện tại.

---

## Bước 3: Huấn luyện cây thứ nhất

Ta huấn luyện một Regression Tree:

$$
h_1(x),
$$

với:

- input: `Height`, `Favorite Color`, `Gender`;
- target: residual $r_{i1}$.

![[Drawing 2026-07-25 22.15.34.excalidraw|1000]]

Ý nghĩa:

- một nhóm mẫu đang bị mô hình dự đoán cao hơn thực tế khoảng $14.7$;
- một nhóm khác đang bị dự đoán thấp hơn thực tế khoảng $4.8$;
- một nhóm cần tăng dự đoán khoảng $16.8$.

---

## Bước 4: Tính giá trị dự đoán tại mỗi lá

Giả sử cây thứ $m$ chia dữ liệu thành các vùng lá:

$$
R_{1m},R_{2m},\dots,R_{J_mm}.
$$

Với mỗi lá $R_{jm}$, ta tìm giá trị cập nhật tốt nhất:

$$
\gamma_{jm}
=
\arg\min_{\gamma}
\sum_{x_i\in R_{jm}}
L
\left(
y_i,
F_{m-1}(x_i)+\gamma
\right).
$$

Với Squared Error, giá trị tối ưu là trung bình residual trong lá:

$$
\gamma_{jm}
=
\frac{1}{|R_{jm}|}
\sum_{x_i\in R_{jm}}
r_{im}.
$$

Nói đơn giản:

> Mỗi lá trả về mức điều chỉnh trung bình mà các mẫu trong lá đang cần.

Ví dụ, nếu một lá chứa hai residual:

$$
-15.2
\quad\text{và}\quad
-14.2,
$$
thì giá trị lá là:

$$
\gamma
=
\frac{-15.2+(-14.2)}{2}
=
-14.7.
$$

Điều này giải thích giá trị $-14.7$ xuất hiện trong cây của ví dụ.

Nếu một mẫu rơi vào lá này, cây đề xuất giảm dự đoán khoảng $14.7$.

---

## Bước 5: Cập nhật mô hình

Sau khi huấn luyện cây thứ nhất, ta cập nhật:

$$
F_1(x)
=
F_0(x)
+
\eta h_1(x).
$$

Trong đó:

- $h_1(x)$: mức điều chỉnh cây thứ nhất đề xuất;
- $\eta$: learning rate.

Giả sử:

$$
F_0(x)=71.2,
$$

$$
h_1(x)=16.8,
$$

và:

$$
\eta=0.1.
$$

Khi đó:

$$
F_1(x)
=
71.2+0.1\times16.8.
$$

$$
F_1(x)
=
72.88.
$$

Nếu không dùng learning rate hoặc $\eta=1$, mô hình sẽ cập nhật toàn bộ:

$$
F_1(x)
=
71.2+16.8
=
88.
$$

Tuy nhiên, cập nhật quá mạnh có thể khiến mô hình dễ overfit. Vì vậy trong thực tế thường sử dụng:

$$
0<\eta<1.
$$

---

## Bước 6: Tính residual mới

Sau khi có mô hình mới $F_1$, ta tính lại residual:

$$
r_{i2}
=
y_i-F_1(x_i).
$$

Residual mới thường nhỏ hơn residual trước đó.

Ví dụ:

$$
y_i=88,
\qquad
F_1(x_i)=72.88.
$$

Khi đó:

$$
r_{i2}
=
88-72.88
=
15.12.
$$

Cây thứ hai sẽ học các residual mới:

$$
\{
(x_i,r_{i2})
\}_{i=1}^{N}.
$$

---

## Bước 7: Huấn luyện cây tiếp theo

Cây thứ hai:

$$
h_2(x),
$$

được huấn luyện để dự đoán residual sau cây thứ nhất.

Sau đó cập nhật:

$$
F_2(x)
=
F_1(x)
+
\eta h_2(x).
$$

Tương tự, cây thứ ba học residual của mô hình $F_2$:

$$
r_{i3}
=
y_i-F_2(x_i).
$$

Sau đó:

$$
F_3(x)
=
F_2(x)
+
\eta h_3(x).
$$

Quá trình được lặp lại đến khi:

- đạt đủ số cây $M$;
- validation loss không còn giảm;
- hoặc sử dụng early stopping.

---

# 7. Công thức tổng quát

Ở vòng $m$, pseudo-residual của mẫu $i$ là:

$$
r_{im}
=
-
\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]_{F=F_{m-1}}.
$$

Huấn luyện cây $h_m(x)$ để dự đoán:

$$
r_{im}.
$$

Cập nhật mô hình:

$$
F_m(x)
=
F_{m-1}(x)
+
\eta h_m(x).
$$

Sau $M$ cây:

$$
\boxed{
F_M(x)
=
F_0(x)
+
\eta
\sum_{m=1}^{M}
h_m(x)
}
$$

Nếu viết rõ theo các lá cây:

$$
F_M(x)
=
F_0(x)
+
\eta
\sum_{m=1}^{M}
\sum_{j=1}^{J_m}
\gamma_{jm}
\mathbb I
\left\{
x\in R_{jm}
\right\}.
$$

Trong đó:

- $J_m$: số lá của cây thứ $m$;
- $R_{jm}$: vùng dữ liệu của lá thứ $j$ trong cây $m$;
- $\gamma_{jm}$: giá trị điều chỉnh tại lá;
- $\mathbb I\{\cdot\}$: hàm chỉ thị.

---

# 8. Dự đoán cho một mẫu mới

Với một mẫu mới $x$, dự đoán được tính như sau:

### Bước 1

Bắt đầu với dự đoán ban đầu:

$$
F_0(x)=\bar y.
$$

### Bước 2

Cho $x$ đi qua cây thứ nhất và lấy giá trị lá:

$$
h_1(x).
$$

Cập nhật:

$$
F_1(x)
=
F_0(x)+\eta h_1(x).
$$

### Bước 3

Cho $x$ đi qua cây thứ hai:

$$
F_2(x)
=
F_1(x)+\eta h_2(x).
$$

### Bước 4

Tiếp tục qua toàn bộ cây:

$$
F_M(x)
=
F_0(x)
+
\eta h_1(x)
+\dots+
\eta h_M(x).
$$

Kết quả cuối cùng:

$$
\widehat y(x)=F_M(x).
$$

---

# 9. Vai trò của learning rate

Learning rate kiểm soát mức đóng góp của mỗi cây:

$$
F_m(x)
=
F_{m-1}(x)
+
\eta h_m(x).
$$

### Learning rate lớn

- mô hình học nhanh;
- cần ít cây;
- mỗi cây sửa lỗi mạnh;
- dễ overfit hơn.

### Learning rate nhỏ

- mỗi cây chỉ sửa một phần nhỏ;
- cần nhiều cây hơn;
- quá trình học ổn định hơn;
- thường có khả năng tổng quát hóa tốt hơn.

Một nguyên tắc thường gặp:

$$
\boxed{
\text{Learning rate nhỏ}
\Longrightarrow
\text{cần nhiều cây hơn}
}
$$

---

# 10. Vì sao cây thường không quá sâu?

Nếu mỗi cây quá sâu, cây có thể học gần như hoàn hảo residual hiện tại.

Điều này có thể làm:

- mô hình sửa lỗi quá mạnh;
- học cả nhiễu trong dữ liệu;
- tăng nguy cơ overfitting.

Gradient Boosting thường sử dụng cây có độ sâu nhỏ, chẳng hạn:
```python
max_depth = 1
max_depth = 2
max_depth = 3
```

### Ưu điểm của Gradient Boosting

* **Độ chính xác rất cao:** Thường đạt hiệu năng dự đoán vượt trội trên dữ liệu dạng bảng (tabular data), dễ dàng đánh bại nhiều thuật toán khác.
* **Hỗ trợ hàm mất mát (Loss Function) linh hoạt:** Có thể tối ưu hóa theo nhiều loại Loss Function khác nhau tùy bài toán (MSE, MAE cho Regression; LogLoss cho Classification; Huber Loss để chống outlier).
* **Không cần chuẩn hóa dữ liệu phức tạp:** Giống các mô hình dựa trên Tree khác, Gradient Boosting không nhạy cảm với quy mô dữ liệu (scaling/normalization) và xử lý tốt dữ liệu phi tuyến.
* **Hỗ trợ xử lý giá trị thiếu (Missing Values):** Nhiều biến thể hiện đại (XGBoost, LightGBM) có cơ chế tự động tìm nhánh phân chia tối ưu cho dữ liệu bị thiếu.
* **Khả năng tùy biến cao:** Cung cấp nhiều kỹ thuật kiểm soát overfitting như Learning Rate (Shrinkage), Subsampling (Bagging), và Regularization ($L_1, L_2$).

---

### Nhược điểm của Gradient Boosting

* **Tốn tài nguyên và thời gian huấn luyện:** Do tính chất huấn luyện tuần tự (cây sau sửa lỗi cây trước), thuật toán khó song song hóa hoàn toàn quá trình dựng cây, dẫn đến thời gian chạy lâu trên tập dữ liệu lớn.
* **Rất nhạy cảm với Hyperparameters:** Hiệu năng phụ thuộc nhiều vào việc tinh chỉnh tham số (learning rate, number of trees, max depth, subsampling). Nếu cấu hình sai rất dễ bị overfitting hoặc underfitting.
* **Khó giải thích (Black-box Nature):** Mặc dù có thể tính toán Feature Importance, việc giải thích chi tiết cách hàng trăm cây phối hợp đưa ra quyết định cho một mẫu cụ thể là rất phức tạp.
* **Nhạy cảm với nhiễu nếu dùng sai Loss Function:** Nếu sử dụng các hàm mất mát như MSE, các điểm dữ liệu nhiễu/outlier vẫn có thể làm lệch đáng kể hướng cập nhật Gradient của mô hình.
* **Tốn bộ nhớ khi lưu trữ mô hình:** Khi số lượng cây lớn (hàng nghìn cây), mô hình chiếm nhiều dung lượng bộ nhớ và làm tăng độ trễ khi chạy Inference.