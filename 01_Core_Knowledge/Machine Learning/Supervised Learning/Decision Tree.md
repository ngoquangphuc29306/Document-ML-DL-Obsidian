---
tags: [ml/algorithm, supervised-learning, decision-tree, classification]
created: 2026-07-21
---

# Decision Tree (Cây quyết định)

## 1. What is Decision Tree?

* **Decision Tree** là một thuật toán thuộc nhóm **[[Types of Machine Learning|Supervised Learning]]**.
* Thuộc nhóm **[[Non-parametric vs Parametric Models|Non-parametric model]]**.
* Hỗ trợ giải quyết cả 2 bài toán: **[[Types of Machine Learning|Classification]]** và **[[Types of Machine Learning|Regression]]**.
* **Bản chất:** Là một chuỗi các quy tắc `IF-ELSE` được sắp xếp theo cấu trúc cây.

### So sánh Rule-based thông thường vs. Decision Tree:
* **Rule-based (Code IF-ELSE thủ công):** Con người tự suy nghĩ ra điều kiện, tự chọn feature và tự đặt ngưỡng (threshold).
* **Decision Tree (Machine Learning):** Máy tự học từ dữ liệu (data), tự chọn feature phù hợp nhất và tự động chọn ngưỡng (threshold) phân chia tối ưu.

---

## 2. Cấu trúc của một Decision Tree

* **Root Node (Nút gốc):** Nút trên cùng chứa toàn bộ tập dữ liệu ban đầu.
* **Internal Node / Branch Node (Nút gian):** Biểu diễn một phép kiểm tra trên một đặc trưng (attribute/feature).
* **Leaf Node / Terminal Node (Nút lá):** Nút cuối cùng biểu diễn kết quả dự đoán (outcomes/classes).
* **Splitting:** Quá trình chia một node cha thành 2 hoặc nhiều node con.
* **Pruning:** Quá trình tỉa bớt (xóa) các nhánh từ cây để tránh hiện tượng [[Overfitting vs Underfitting|Overfitting]].
![[Pasted image 20260721111640.png]]
---

# <font color = "red">Classification in Decision Tree</font>

## 1. Thuật toán tổng quát (Tree Theory)

* **Step 1:** Xây dựng cây với nút gốc (kí hiệu $S$) chứa toàn bộ dataset.
* **Step 2:** Tìm thuộc tính (attribute) tốt nhất trong dataset bằng thước đo chọn thuộc tính **Attribute Selection Measure (ASM)**.
* **Step 3:** Split dữ liệu thành các tập con dựa trên các giá trị có thể của thuộc tính tốt nhất đó.
* **Step 4:** Tạo decision node chứa thuộc tính tốt nhất.
* **Step 5:** Tiếp tục xây dựng decision tree bằng đệ quy (**Recursion**) với các sub-dataset được tạo ra ở Step 3. Dừng lại khi không thể phân chia được nữa (hoặc đạt điều kiện dừng).

---

## 2. Purity và Impurity

* **Impurity (Mức độ tạp chất/hỗn loạn):** Đo lường mức độ trộn lẫn (diversity) giữa các lớp (classes) trong cùng một node.
  * **Pure Node (Purity cao):** Node chứa toàn bộ (hoặc đa số) các mẫu thuộc về **cùng 1 class** $\rightarrow$ Dễ quyết định.
  * **Impure Node (Impurity cao):** Node chứa hỗn hợp nhiều class khác nhau với tỷ lệ tương đương $\rightarrow$ Khó quyết định.

### Dataset mẫu để tính toán:
Xét bài toán dự đoán xem một người có **Love AI** (Yes/No) hay không:

| ID | Love Math | Love Art | Age | Love AI (Label) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | Yes | Yes | 7 | No |
| 2 | No | No | 12 | No |
| 3 | Yes | Yes | 18 | Yes |
| 4 | Yes | Yes | 35 | Yes |
| 5 | Yes | Yes | 38 | Yes |
| 6 | No | No | 50 | No |
| 7 | No | No | 83 | No |

**Tổng quan Dataset ($S$):** 7 mẫu (3 mẫu `Yes`, 4 mẫu `No`).

---

## 3. Evaluation Metrics

Để chọn thuộc tính tốt nhất làm nút phân chia, ta sử dụng 2 chỉ số chính: **Gini Impurity** và **Entropy (Information Gain)**.
### a) Gini Impurity

Công thức tính Gini Impurity cho nút $D_i$ có $k$ lớp:

$$Gini(D_i) = 1 - \sum_{j=1}^{k} p_j^2$$

* $p_j$: Tỷ lệ mẫu thuộc class $j$ trong node $D_i$.
* $Gini = 0 \Rightarrow$ Node hoàn toàn tinh khiết (**Pure**).
* **Với bài toán Phân loại 2 lớp (Binary Classification):**
  * $Gini_{\max} = 0.5$ khi tỷ lệ $p = 0.5$ cho mỗi class.
* **Với $n$ lớp bất kỳ (General Case):**
  $$Gini_{\max} = 1 - \sum_{i=1}^{n} \left(\frac{1}{n}\right)^2 = 1 - \frac{1}{n} = \frac{n-1}{n}$$

#### Gini của phép chia (Weighted Gini of Split):
$$Gini_{split}(D) = \sum_{i=1}^{m} \frac{|S_i|}{|S|} \cdot Gini(D_i) = \sum_{i=1}^{m} \frac{\text{count}(S_i)}{\text{count}(S)} \cdot Gini(D_i)$$

---

#### Ví dụ 1: Tính Gini cho Categorical Feature

##### **1. Phép chia theo `Love Math`:**
* **`Love Math = Yes`** $\rightarrow$ 1 `Yes`, 3 `No`:
  $$Gini(\text{LoveMath}_1) = 1 - \left(\frac{3}{4}\right)^2 - \left(\frac{1}{4}\right)^2 = 1 - \frac{9}{16} - \frac{1}{16} = 0.375$$
* **`Love Math = No`** (3 mẫu: ID 2, 6, 7) $\rightarrow$ 2 `Yes`, 1 `No`:
  $$Gini(\text{LoveMath}_2) = 1 - \left(\frac{2}{3}\right)^2 - \left(\frac{1}{3}\right)^2 = 0.444$$
* **Gini tổng thể cho `Love Math`:**
  $$Gini(\text{Love Math}) = \frac{4}{7} \cdot 0.375 + \frac{3}{7} \cdot 0.444 = 0.405$$
##### **2. Phép chia theo `Love Art`:**
* **`Love Art = Yes`** (4 mẫu: ID 1, 3, 4, 5) $\rightarrow$ 3 `Yes`, 1 `No`:
  $$Gini(\text{LoveArt}_1) = 1 - \left(\frac{3}{4}\right)^2 - \left(\frac{1}{4}\right)^2 = 0.375$$
* **`Love Art = No`** (3 mẫu: ID 2, 6, 7) $\rightarrow$ 0 `Yes`, 3 `No`:
  $$Gini(\text{LoveArt}_2) = 1 - \left(\frac{0}{3}\right)^2 - \left(\frac{3}{3}\right)^2 = 0$$
* **Gini tổng thể cho `Love Art`:**
  $$Gini(\text{Love Art}) = \frac{4}{7} \cdot 0.375 + \frac{3}{7} \cdot 0 = 0.214$$
---

#### Ví dụ 2: Xử lý Numerical Feature (`Age`)

Khi gặp đặc trưng dạng số, ta thực hiện các bước sau:
1. **Sắp xếp (Sort)** dữ liệu theo cột `Age` tăng dần.
2. **Tính giá trị trung bình (Average)** của các cặp điểm liên tiếp để làm các điểm ngưỡng **Threshold** tiềm năng.

Các điểm Threshold cho `Age`:
![[Pasted image 20260721112830.png]]
##### Tính Gini cho từng Threshold:
* **Thử $Age < 9.5$:**
  * True (1 mẫu: `No`=1, `Yes`=0) $\rightarrow Gini = 0$
  * False (6 mẫu: `No`=3, `Yes`=3) $\rightarrow Gini = 1 - (3/6)^2 - (3/6)^2 = 0.5$
  $$\Rightarrow Gini(Age < 9.5) = \frac{1}{7} \cdot 0 + \frac{6}{7} \cdot 0.5 = 0.429$$

* **Thử $Age < 15$:**
  * True (2 mẫu: `No`=2, `Yes`=0) $\rightarrow Gini = 0$
  * False (5 mẫu: `No`=2, `Yes`=3) $\rightarrow Gini = 1 - (2/5)^2 - (3/5)^2 = 0.48$
  $$\Rightarrow Gini(Age < 15) = \frac{2}{7} \cdot 0 + \frac{5}{7} \cdot 0.48 = 0.343$$
##### Kết quả so sánh Gini cho toàn bộ thuộc tính:
* $Gini(\text{Dataset gốc}) = 1 - \left(\frac{4}{7}\right)^2 - \left(\frac{3}{7}\right)^2 = 0.489$
* $Gini(Age < 15) = 0.343$
* $Gini(\text{Love Art}) = 0.214$
* $Gini(\text{Love Math}) = 0.405$

> **Quy tắc chọn:** Chọn phép chia có **Gini nhỏ nhất** và nhỏ hơn $Gini$ của node cha ($0.489$).  
> $\rightarrow$ Chọn **`Love Art`** với $Gini = 0.214$ làm nút gốc!

---

### b) Entropy & Information Gain

#### Bản chất khái niệm Surprise (Sự bất ngờ) & Entropy:
* **Mối quan hệ nghịch biến (Inverse Relationship):** Xác suất $P$ càng nhỏ thì mức độ bất ngờ (Surprise) càng lớn.
  $$Surprise = \frac{1}{P}$$
* Vấn đề là khi trong một set chỉ toàn một class thì P = 1 $\rightarrow$ S = 1 (vô lý). Nên ta phải scale lại bằng $\log_2()$.
  $$Surprise = \log_2\left(\frac{1}{P}\right) = -\log_2(P)$$
  * Nhận sự kiện có $P = 0.9 \Rightarrow Surprise = -\log_2(0.9) \approx 0.15$ (Không bất ngờ).
  * Nhận sự kiện có $P = 0.1 \Rightarrow Surprise = -\log_2(0.1) \approx 3.32$ (Rất bất ngờ!).
  * Nhận sự kiện có $P = 0 \Rightarrow Surprise = -\log_2(0) \rightarrow +\infty$ (Không bất ngờ). 
![[SurpriseProbability.excalidraw|1000]]
* **Entropy (total suprise):** là mức độ lẫn lộn hoặc không chắc chắc (uncertainly) của node.

$$Entropy(S) = -\sum_{i=1}^{K} p_i \cdot \log_2(p_i)$$

* **Với Binary Classification ($n=2$):** $Entropy_{\max} = 1$ khi $p = 0.5$.
![[Pasted image 20260721113739.png]]
* **Với $n$ class bất kỳ:**
  $$Entropy_{\max} = - \sum_{i=1}^{n} \frac{1}{n} \log_2\left(\frac{1}{n}\right) = -\log_2\left(\frac{1}{n}\right)$$
![[Pasted image 20260721114649.png]]
---

#### Information Gain (IG - Độ lợi thông tin):
**Information Gain** đo lường mức độ giảm Entropy (độ hỗn loạn) sau khi chia dữ liệu theo một thuộc tính $F$. Lượng thông tin thu được càng lớn càng tốt.

$$IG(S, F) = Entropy(S) - \sum_{v \in Values(F)} \frac{|S_v|}{|S|} \cdot Entropy(S_v)$$

> **Quy tắc:** Chọn thuộc tính có **Information Gain lớn nhất (Max IG)** làm nút chia.

---

#### Ví dụ tính toán Information Gain trên Dataset:
![[IG_1.excalidraw|500]] ![[IG_2.excalidraw|500]]
1. **Entropy tại Node Cha (Entire Population $S$):**
   * Tập $S$ gồm 3 `Yes` và 4 `No`.
   $$Entropy(S) = -\frac{3}{7} \log_2\left(\frac{3}{7}\right) - \frac{4}{7} \log_2\left(\frac{4}{7}\right) = 0.985$$
2. **Nếu chia theo `Love Art`:**
   * Branch `Love Art = Yes` (4 mẫu: 3 `Yes`, 1 `No`):
     $$Entropy(S_{f1}) = -\frac{3}{4} \log_2\left(\frac{3}{4}\right) - \frac{1}{4} \log_2\left(\frac{1}{4}\right) = 0.811$$
   * Branch `Love Art = No` (3 mẫu: 0 `Yes`, 3 `No`):
     $$Entropy(S_{f2}) = -\frac{3}{3} \log_2\left(\frac{3}{3}\right) = 0$$
   * **Information Gain:**
     $$IG(S, \text{Love Art}) = 0.985 - \left(\frac{4}{7} \cdot 0.811 + \frac{3}{7} \cdot 0\right) = 0.522$$

3. **Nếu chia theo `Love Math`:**
   * Branch `Love Math = Yes` (4 mẫu: 3 `Yes`, 1 `No`) $\rightarrow Entropy = 0.811$.
   * Branch `Love Math = No` (3 mẫu: 1 `Yes`, 2 `No`):
     $$Entropy = -\frac{1}{3} \log_2\left(\frac{1}{3}\right) - \frac{2}{3} \log_2\left(\frac{2}{3}\right) = 0.918$$
   * **Information Gain:**
     $$IG(S, \text{Love Math}) = 0.985 - \left(\frac{4}{7} \cdot 0.811 + \frac{3}{7} \cdot 0.918\right) = 0.129$$

> **Kết luận:** So sánh $IG(\text{Love Art}) = 0.522$ vs $IG(\text{Love Math}) = 0.129$.  
> Thuộc tính **`Love Art`** đem lại Information Gain cao nhất $\rightarrow$ Chọn **`Love Art` làm nút đầu tiên**! Tiếp tục mở rộng cây cho đến khi không thể chia được nữa.

# <font color = "red">Regression in Decision Tree</font>

## 1. Motivation
- Hạn chế của [[Linear Regression]] chỉ mô hình hóa tốt các mối quan linear. Khi data có dạng non-linear thì sẽ bị [[Bias-Variance Tradeoff|High Bias]] (Underfitting).
## 2. Attribute Selection Measure
- Để tìm nút phân chia tốt nhất cho **Regression Tree**, ta đánh giá mức phân tác của biến mục tiêu liên tục ở các nốt sau split.
### 2.1. Mean Squared Error và Sum of Squared Errors

Trong Regression Tree, giá trị dự đoán tại một nút hoặc một nút lá thường là giá trị trung bình của các target thuộc nút đó.

Với nút $D_i$ chứa $n_i$ mẫu, giá trị dự đoán tại nút là:

$$
\widehat{y}_{D_i}
=
\overline{y}_{D_i}
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}y_j.
$$

#### Mean Squared Error của một nút

**Mean Squared Error (MSE)** đo trung bình bình phương sai lệch giữa target thật và giá trị dự đoán tại nút:

$$
MSE(D_i)
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\left(
y_j-\overline{y}_{D_i}
\right)^2.
$$

#### Sum of Squared Errors của một nút

**Sum of Squared Errors (SSE)** là tổng bình phương sai lệch tại nút:

$$
SSE(D_i)
=
\sum_{j=1}^{n_i}
\left(
y_j-\overline{y}_{D_i}
\right)^2.
$$

Mối liên hệ giữa MSE và SSE là:

$$
SSE(D_i)
=
n_i \cdot MSE(D_i).
$$

Trong đó:

- $n_i$: số lượng mẫu tại nút $D_i$;
- $y_j$: target thật của mẫu thứ $j$;
- $\overline{y}_{D_i}$: trung bình target của các mẫu trong nút $D_i$;
- giá trị dự đoán của nút lá chính là $\overline{y}_{D_i}$.

> [!important]
> Trong Regression Tree, nên sử dụng thống nhất thuật ngữ **SSE** hoặc **RSS** cho tổng bình phương phần dư.  
> Không nên dùng `SSR` vì ký hiệu này trong một số tài liệu được dùng cho Regression Sum of Squares.
### 2.2. Quy tắc chọn phép chia tốt nhất

Mục tiêu của Regression Tree là tìm feature và threshold sao cho các target trong từng nút con càng ít phân tán càng tốt.
Giả sử một phép chia tạo ra hai nút con:
$$
D_L
=
\{(x_i,y_i):x_i\le s\},
$$
$$
D_R
=
\{(x_i,y_i):x_i>s\}.
$$

Có hai cách tương đương để đánh giá một phép chia.
#### Cách 1: Tổng SSE của các nút con
$$
SSE_{\text{split}}
=
SSE(D_L)
+
SSE(D_R).
$$
Cụ thể:
$$
SSE_{\text{split}}
=
\sum_{x_i\in D_L}
\left(
y_i-\overline{y}_L
\right)^2
+
\sum_{x_i\in D_R}
\left(
y_i-\overline{y}_R
\right)^2.
$$
Trong đó:
$$
\overline{y}_L
=
\frac{1}{|D_L|}
\sum_{x_i\in D_L}y_i,
$$
$$
\overline{y}_R
=
\frac{1}{|D_R|}
\sum_{x_i\in D_R}y_i.
$$
#### Cách 2: Weighted MSE
Nếu sử dụng MSE, cần lấy trung bình có trọng số theo số lượng mẫu của từng nút con:

$$
MSE_{\text{split}}
=
\frac{|D_L|}{|D|}
MSE(D_L)
+
\frac{|D_R|}{|D|}
MSE(D_R).
$$

Hai tiêu chí này cho cùng thứ tự lựa chọn phép chia vì:
$$
SSE_{\text{split}}
=
|D|\cdot MSE_{\text{split}}.
$$

> [!warning]
> Không nên sử dụng:
>
> $$
> MSE(D_L)+MSE(D_R)
> $$
>
> vì cách cộng này xem hai nút con có mức ảnh hưởng ngang nhau, ngay cả khi số lượng mẫu của chúng rất khác nhau.

#### Đối với Numerical Feature
1. **Sắp xếp dữ liệu:**  
   Sắp xếp các mẫu theo giá trị của feature theo thứ tự tăng dần.
2. **Tạo threshold ứng viên:**  
   Với hai giá trị liên tiếp khác nhau $x_{(j)}<x_{(j+1)}$, tạo threshold:
   $$
   s_j
   =
   \frac{x_{(j)}+x_{(j+1)}}{2}.
   $$
3. **Chia dữ liệu:**  
   Với mỗi threshold $s_j$, tạo hai tập:
   $$
   D_L
   =
   \{(x_i,y_i):x_i\le s_j\},
   $$
$$
   D_R
   =
   \{(x_i,y_i):x_i>s_j\}.
   $$
4. **Tính tổng sai số sau phép chia:**
   $$
   SSE_{\text{split}}(s_j)
   =
   SSE(D_L)+SSE(D_R).
   $$
5. **Chọn threshold tốt nhất:**
$$
   s^*
   =
   \arg\min_{s_j}
   SSE_{\text{split}}(s_j).
   $$

Phép chia có tổng SSE nhỏ nhất được xem là phép chia tốt nhất đối với feature đang xét.
## 3. Overfitting and Solution
Tham khảo lý thuyết tại: [[Overfitting vs Underfitting]]
- Nếu cho phép câu phát triển tự do, cây sẽ tiếp tục chia nhanh cho tới khi mỗi nút là chỉ chứa 1 hoặc rất ít mẫu dữ liệu (khi đó $SSE=0$ trên tập **Train**).
- Điều này dẫn đến hiện trượng [[Bias-Variance Tradeoff|High Variance]] (Overfitting).
### 3.1. Solution 1: Stopping Early (Pre-Pruning)
Dừng quá trình phân nhánh sớm trước khi cây trở nên quá sâu dựa trên các siêu tham số (hyperparameters):
* **Maximim Depth** (`max_depth`): Dừng chia nhánh khi cây đạt đến độ sâu nhất định.
* **Minimum Samples per Leaf** (`min_samples_leaf`): Một nút chỉ được coi là "lá" nếu nó chứa ít nhất $k$ mẫu dữ liệu. Nếu việc chia tiếp tạo ra nút con có số mẫu $< k$, phép chia sẽ bị hủy.
* **Minimum Samples Split** (`min_samples_split`): Một nút nội (internal node) phải có ít nhất $N$ mẫu thì mới được phép xem xét chia nhỏ tiếp.
* **Minimum Information Gain / Impurity Decrease** (`min_impurity_decrease`): Dừng chia nhánh nếu phép chia tốt nhất không giúp giảm độ không tinh khiết (Gini/Entropy/MSE) vượt qua một ngưỡng $Threshold$ cho trước.
* **Maximum Leaf Nodes** (`max_leaf_nodes`): Giới hạn tổng số nút lá tối đa mà toàn bộ cây có thể sở hữu. Cây sẽ ưu tiên mở rộng các nhánh làm giảm độ không tinh khiết nhiều nhất trước.
#### Python Code:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(max_depth=10):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    train_recall = recall_score(y_train, y_pred_train)
    test_recall = recall_score(y_test, y_pred_test)

    return train_accuracy, test_accuracy, train_recall, test_recall, clf

max_depths = []
train_accuracies = []
test_accuracies = []
train_recalls = []
test_recalls = []
classifiers = []
for i in range(1, 11):
    train_accuracy, test_accuracy, train_recall, test_recall, clf = train_and_evaluate(max_depth=i)
    
	train_accuracies.append(train_accuracy)
	test_accuracies.append(test_accuracy)
    train_recalls.append(train_recall)
    test_recalls.append(test_recall)
    max_depths.append(i)
    classifiers.append(clf)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(max_depths, train_accuracies, label='Training Accuracy')
plt.plot(max_depths, test_accuracies, label='Test Accuracy')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Maximum Depth')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(max_depths, train_recalls, label='Training Recall')
plt.plot(max_depths, test_recalls, label='Test Recall')
plt.xlabel('Maximum Depth')
plt.ylabel('Recall')
plt.title('Recall vs Maximum Depth')
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_recall_plot.webp")

plt.show()
```

---

### 3.2. Solution 2: Post-Pruning (Cost Complexity Pruning)
Xây dựng một cây đầy đủ trước (Full-size Tree), sau đó tỉa bớt các nhánh dựa trên hàm phạt độ phức tạp.

**Công thức Cost-Complexity (Tree Score):**
$$R_\alpha(T) = R(T) + \alpha \cdot |T|$$

* **$R(T)$ (Độ đo sai số của cây):**
  * **Classification:** Tỷ lệ lỗi phân loại (Misclassification Rate) hoặc tổng độ không tinh khiết (Gini Impurity / Entropy) tại các lá.
  * **Regression:** Tổng bình phương phần dư ($\text{Total SSR} = \sum (y_i - \hat{y})^2$) tại các lá.
* **$|T|$ (hay $T$):** Tổng số nút lá (Terminal nodes / Leaves).
* **$\alpha$ (Alpha):** Hệ số phạt độ phức tạp (tuning parameter).
  * Khi $\alpha = 0$: Cây full-size có $R_\alpha(T)$ nhỏ nhất trên tập Train (Dễ Overfit).
  * Khi $\alpha$ tăng: Phạt mạnh các cây nhiều lá $\rightarrow$ Ép cây phải gọn nhẹ hơn.

**Quy trình chọn $\alpha$ tối ưu bằng $K$-Fold Cross-Validation:**
1. Chia tập Train thành $K$ folds.
2. Xây dựng chuỗi các cây con ứng với các giá trị $\alpha$ tiềm năng ($\alpha_0 < \alpha_1 < \dots < \alpha_m$).
3. Với mỗi $\alpha$, đánh giá sai số trung bình trên tập Validation:
   * **Classification:** Đo tỷ lệ lỗi phân loại (Validation Classification Error).
   * **Regression:** Đo tổng bình phương phần dư (Validation SSR / MSE).
1. Chọn $\alpha_{opt}$ có lỗi Validation trung bình thấp nhất để tiến hành tỉa cây chính thức.
![[Pasted image 20260721205911.png]]

#### Python Code:
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Generate synthetic data
X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, random_state=42
)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# BƯỚC 1: Khởi tạo, fit cây ban đầu và lấy danh sách ccp_alphas
base_clf = DecisionTreeClassifier(random_state=42)
base_clf.fit(X_train, y_train)

path = base_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# BƯỚC 2: Dùng Cross-Validation tìm alpha tốt nhất
scores_cv = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    scores_cv.append(scores.mean())

# Tìm index của alpha có score cao nhất
best_idx = np.argmax(scores_cv)
best_alpha = ccp_alphas[best_idx]
best_cv_accuracy = scores_cv[best_idx]

# Vẽ biểu đồ CV Score vs Alpha
plt.figure(figsize=(8, 5))
plt.plot(
    ccp_alphas[:-1],
    scores_cv[:-1],
    marker="o",
    color="b",
    label="CV Mean Accuracy",
)
plt.axvline(
    x=best_alpha, color="r", linestyle="--", label=f"Best Alpha ({best_alpha:.4f})"
)
plt.xlabel("Alpha (ccp_alpha)")
plt.ylabel("Mean Validation Accuracy")
plt.title("Cross-Validation Accuracy vs Alpha")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# BƯỚC 3: Chọn Alpha tối ưu và fit đúng 1 cây cuối cùng
best_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
best_clf.fit(X_train, y_train)

# Đánh giá cuối cùng trên tập Test
test_accuracy = best_clf.score(X_test, y_test)

print(f"Best alpha tìm được: {best_alpha:.6f}")
print(f"Best CV accuracy (trên tập Train): {best_cv_accuracy * 100:.2f}%")
print(f"Test Accuracy (trên tập Test độc lập): {test_accuracy * 100:.2f}%")
```

---
# <font color = "red">Advantages and Disadvantages</font>

- **Advantages**:
	- Very easy to explain.
	- More closely mirror human.
	- Can easily handel qualitative predictors without the need of create **dummy** variables.
- **Disadvantages**:
	- Do not have the same level of predicting accuracy as some other regression and classification methods.
	- Small changes in the data can cause a large change in the large estimated tree.
	- Are less effective in making predictions when the main goal is to predict the outcomes of a continuous variable.
# <font color = "red">Implementaion Sklearn</font>

## 1. Classification

### 1.1 Train with Entropy/Gini
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Generate synthetic data
X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, random_state=42
)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Huấn luyện Decision Tree với Entropy
clf = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

clf.fit(X_train, y_train)

# Dự đoán
y_pred = clf.predict(X_test)

# Đánh giá
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
```

### 1.2 Train with cross-validation
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Generate synthetic data
X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, random_state=42
)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# BƯỚC 1: Khởi tạo, fit cây ban đầu và lấy danh sách ccp_alphas
base_clf = DecisionTreeClassifier(random_state=42)
base_clf.fit(X_train, y_train)

path = base_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# BƯỚC 2: Dùng Cross-Validation tìm alpha tốt nhất
scores_cv = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    scores_cv.append(scores.mean())

# Tìm index của alpha có score cao nhất
best_idx = np.argmax(scores_cv)
best_alpha = ccp_alphas[best_idx]
best_cv_accuracy = scores_cv[best_idx]

# Vẽ biểu đồ CV Score vs Alpha
plt.figure(figsize=(8, 5))
plt.plot(
    ccp_alphas[:-1],
    scores_cv[:-1],
    marker="o",
    color="b",
    label="CV Mean Accuracy",
)
plt.axvline(
    x=best_alpha, color="r", linestyle="--", label=f"Best Alpha ({best_alpha:.4f})"
)
plt.xlabel("Alpha (ccp_alpha)")
plt.ylabel("Mean Validation Accuracy")
plt.title("Cross-Validation Accuracy vs Alpha")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# BƯỚC 3: Chọn Alpha tối ưu và fit đúng 1 cây cuối cùng
best_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
best_clf.fit(X_train, y_train)

# Đánh giá cuối cùng trên tập Test
test_accuracy = best_clf.score(X_test, y_test)

print(f"Best alpha tìm được: {best_alpha:.6f}")
print(f"Best CV accuracy (trên tập Train): {best_cv_accuracy * 100:.2f}%")
print(f"Test Accuracy (trên tập Test độc lập): {test_accuracy * 100:.2f}%")
```

## 2. Regression

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

# 1. Tạo dữ liệu giả lập cho bài toán Regression
X, y = make_regression(
    n_samples=1000, n_features=10, noise=15, random_state=42
)

# Chia tập Train và Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# BƯỚC 1: Khởi tạo, fit cây gốc và lấy danh sách ccp_alphas
base_reg = DecisionTreeRegressor(random_state=42)
base_reg.fit(X_train, y_train)

path = base_reg.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# BƯỚC 2: Dùng 5-Fold Cross-Validation để tìm Alpha tối ưu dựa trên MSE
# Lưu ý: cross_val_score dùng 'neg_mean_squared_error' nên cần đảo dấu (-)
cv_mse_scores = []

for ccp_alpha in ccp_alphas:
    reg = DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alpha)
    # Lấy giá trị âm của MSE từ cross_val_score rồi đổi dấu thành dương
    neg_mse = cross_val_score(
        reg, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    cv_mse_scores.append(-neg_mse.mean())

# Tìm Alpha làm cho MSE trên Cross-Validation là NHỎ NHẤT
best_idx = np.argmin(cv_mse_scores)
best_alpha = ccp_alphas[best_idx]
best_cv_mse = cv_mse_scores[best_idx]

# Vẽ biểu đồ CV MSE vs Alpha
plt.figure(figsize=(8, 5))
plt.plot(
    ccp_alphas[:-1],
    cv_mse_scores[:-1],
    marker="o",
    color="orange",
    label="CV Mean MSE",
)
plt.axvline(
    x=best_alpha, color="r", linestyle="--", label=f"Best Alpha ({best_alpha:.4f})"
)
plt.xlabel("Alpha (ccp_alpha)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Cross-Validation MSE vs Alpha (Regression)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# BƯỚC 3: Fit cây Regressor tối ưu với best_alpha tìm được
best_reg = DecisionTreeRegressor(random_state=42, ccp_alpha=best_alpha)
best_reg.fit(X_train, y_train)

# Đánh giá trên tập Test
y_pred_test = best_reg.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"👉 Best Alpha tìm được: {best_alpha:.6f}")
print(f"👉 Best CV MSE (trên tập Train): {best_cv_mse:.2f}")
print(f"🎯 Test MSE (trên tập Test độc lập): {test_mse:.2f}")
print(f"📊 Test R² Score: {test_r2 * 100:.2f}%")
```