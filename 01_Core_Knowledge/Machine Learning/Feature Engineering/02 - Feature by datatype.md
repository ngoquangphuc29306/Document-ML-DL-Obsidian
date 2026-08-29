---
title: "Feature Engineering theo loại dữ liệu"
tags: [feature-engineering, numerical, categorical, text]
---
# Phần 2 — Feature Engineering theo loại dữ liệu

## 1. Dữ liệu số

### Log transform
Dùng với phân phối lệch phải (right-skewed):
$$x' = \log(1+x)$$

*Phù hợp với doanh thu, lượt xem, giá trị giao dịch.*

---

### Ratio features
$$\text{Debt-to-Income} = \frac{\text{Debt}}{\text{Income}}$$

$$\text{Area per Room} = \frac{\text{Area}}{\text{Rooms}}$$

> [!warning] Warning
> Luôn kiểm tra và xử lý trường hợp mẫu số bằng 0 (division by zero).

---

### Interaction features
$$x_{\text{interaction}} = x_1 x_2$$

*Linear model* thường cần tạo interaction rõ ràng; *Tree-based model* có thể tự học nhưng feature nghiệp vụ được tạo sẵn vẫn rất có ích.

---

### Binning
Chuyển giá trị liên tục thành nhóm:
* **Age:** `0–17`, `18–24`, `25–34`, `35–49`, `50+`

> [!note] Note
> Binning giúp mô hình dễ giải thích hơn nhưng sẽ làm mất đi thông tin chi tiết của dữ liệu liên tục.

---

### Aggregation
Gom nhóm theo khách hàng hoặc sản phẩm để tính:
* Count
* Sum
* Mean
* Median
* Min / Max
* Standard deviation
* Tỷ lệ lỗi
* Tỷ lệ hoàn trả

---

## 2. Dữ liệu phân loại (Categorical Data)

### One-hot encoding
Phù hợp với category **không có thứ tự** và **cardinality thấp** (ít giá trị phân biệt).

---

### Ordinal encoding
Chỉ dùng khi category có thứ tự thực sự:
$$\text{low} < \text{medium} < \text{high}$$

---

### Frequency encoding
$$FE(c) = \frac{\operatorname{count}(c)}{N}$$

*Hữu ích với category có cardinality cao.*

---

### Target encoding
$$TE(c) = E[y \mid c]$$

> [!danger] Rủi ro Leakage Cao
> Target encoding rất dễ gây **Data Leakage**. 
> * **Bắt buộc:** Dùng out-of-fold encoding, smoothing và **chỉ fit trên tập train**.
> 
> **Công thức Smoothing:**
> $$TE(c) = \frac{n_c \bar{y}_c + \alpha \mu}{n_c + \alpha}$$
> *Trong đó $n_c$ là số mẫu của category $c$, $\bar{y}_c$ là mean target của $c$, $\mu$ là global mean, và $\alpha$ là hệ số smoothing.*

---

### Rare categories
Gộp các category hiếm xuất hiện thành nhóm **`Other`** để giảm nhiễu và xử lý category mới (unseen categories) khi inference ổn định hơn.

---

## 3. Dữ liệu văn bản (Text Data)

### Các feature cơ bản
* Số ký tự
* Số từ
* Số câu
* Tỷ lệ viết hoa
* Số URL
* Số dấu chấm than

---

### Biểu diễn ngữ nghĩa (Representation)
* **TF-IDF:**
  $$TFIDF(t, d) = TF(t, d) \times IDF(t)$$
* **N-grams**
* **Embeddings**

> [!note] Note
> Không phải lúc nào cũng nên xóa emoji, dấu câu hoặc chữ viết hoa; chúng có thể chứa tín hiệu cảm xúc (sentiment) rất quan trọng.