---
tags: [ml/algorithm, supervised-learning, classification, naive-bayes, probability]
created: 2026-07-22
---

# Naive Bayes Classifier (Bộ phân loại Bayes ngây thơ)

**Naive Bayes** là một họ thuật toán học có giám sát (Supervised Learning) dựa trên **Định lý Bayes (Bayes' Theorem)** với một giả định "ngây thơ" (*naive*): **các đặc trưng (features) độc lập điều kiện với nhau** khi biết nhãn lớp (class label).

---

## 1. Cơ sở toán học (Mathematical Foundation)

### 1.1. Định lý Bayes (Bayes' Theorem)

Xét biến mục tiêu $y$ (lớp/nhãn) và vectơ đặc trưng $X = (x_1, x_2, \dots, x_n)$:

$$P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)}$$

* $P(y \mid X)$ (**Posterior Probability**): Xác suất hậu kỳ (xác suất thuộc lớp $y$ khi đã biết tập đặc trưng $X$).
* $P(X \mid y)$ (**Likelihood**): Khả năng xảy ra tập đặc trưng $X$ khi đã biết thuộc lớp $y$.
* $P(y)$ (**Prior Probability**): Xác suất tiên kỳ (tỷ lệ xuất hiện của lớp $y$ trong toàn bộ tập dữ liệu).
* $P(X)$ (**Evidence**): Xác suất xuất hiện của tập đặc trưng $X$ (đóng vai trò hằng số chuẩn hóa).

### 1.2. Giả định

Giả định rằng các đặc trưng $x_1, x_2, \dots, x_n$ độc lập với nhau khi biết $y$:

$$P(X \mid y) = P(x_1, x_2, \dots, x_n \mid y) = \prod_{i=1}^{n} P(x_i \mid y)$$

### 1.3. Công thức ra quyết định (MAP - Maximum A Posteriori)

Vì $P(X)$ giống nhau đối với mọi lớp $y$, ta bỏ qua mẫu số $P(X)$ và tìm lớp $y$ làm cực đại hóa xác suất:

$$\hat{y} = \arg\max_{y} \left( P(y) \cdot \prod_{i=1}^{n} P(x_i \mid y) \right)$$

---

## 2. Các biến thể phổ biến của Naive Bayes

Mỗi biến thể được thiết kế dựa trên bản chất phân bố dữ liệu của các đặc trưng $x_i$:

### 2.1. Gaussian Naive Bayes
* **Dùng cho:** Đặc trưng dạng **số liên tục (Continuous Features)** tuân theo phân bố chuẩn (Gaussian Distribution).
* **Công thức Likelihood:**
  $$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left( -\frac{(x_i - \mu_y)^2}{2\sigma_y^2} \right)$$
  *(Trong đó $\mu_y$ và $\sigma_y^2$ là trung bình và phương sai của đặc trưng $x_i$ thuộc lớp $y$)*
![[Pasted image 20260722091405.png|697]]
- **Code Python:**
```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.Nave_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["Species"] = iris.target

X = data.drop("Species", axis=1)
y = data["Species"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
gnb = GaussianNB()

gnb.fit(X_train, y_train)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

feature_names = iris.feature_names
num_features = len(feature_names)
num_classes = len(np.unique(y))

X_np = X.to_numpy()

for feature_index in range(num_features):
	feature_name = feature_names[feature_index]
	x_vals = np.linespace(X_np[:, feature_index].min(),
						  X_np[:, feature_index].max(), 200)

plt.figure(figsize=(8, 4))
for cls in range(num_classes):
	mean = gnb.theta_[cls, feature_index]
	std = np.sqrt(gnb.var_[cls, feature_index])
	y_vals = norm.pdf(x_vals, mean, std)
	plt.plot(x_vals, y_vals, label=f"Class {cls} ({iris.target_names[cls]})")
	
plt.title(f"Gaussian Distribution - {feature_name}")
plt.xlabel(feature_name)
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"The Accuracy of Prediction on Iris Flower is: {accuracy}")
```

---
### 2.2. Multinomial Naive Bayes
* **Dùng cho:** Dữ liệu dạng **đếm tần suất (Discrete Counts)**, rất phổ biến trong Xử lý ngôn ngữ tự nhiên (NLP) như Bag-of-Words, TF-IDF.
* Phương pháp này hoạt động bằng cách sử dụng số lượng từ để phân loại văn bản. Ý tưởng chính là giả định rằng môi từ trong môt thông điệp hoặc đặc trưng là độc lập với nhau. Điều này có nghĩa là sự hiện diện của một từ không ảnh hưởng đến sự hiện diện của một từ khác, giúp mô hình dễ sử dụng
* Mô hình này xem xét tần suất xuất hiện của mỗi từ trong các tin nhắn thuộc các danh mục khác nhau (như “thư rác” hoặc “không phải thư rác”). Ví dụ: nếu từ “miễn phí” xuất hiện thường xuyên trong tin nhắn thư rác, điều này sẽ giúp dự đoán liệu một tin nhắn mới có phải là thư rác hay không.
* Để tính xác xuất của một thông điệp thuộc về một danh mục nhát định, **Multinomial Naive Bayes** sử dụng phân phối đa thức (multinomial distribution) như sau:
$$P(X) = \frac{n!}{n_1! \cdot n_2! \cdot \ldots \cdot n_m!} \cdot p_1^{n_1} \cdot p_2^{n_2} \cdot \ldots \cdot p_m^{n_m}$$
- Giải thích các tham số:
	- n :  tổng số lần thử nghiệm
	- ${n}_i$ : số lần xuất hiện của kết quả i
	- ${p}_i$ : xác xuất của kết quả
- Để ước tính khả năng mỗi từ thuộc một nhóm cụ thể như **“spam”** hay **“không phải spam”**, chúng tôi sử dụng phương pháp **Maximum Likelihood Estimation (MLE)**. Phương pháp này giúp tìm ra xác suất dựa trên số lượng thực tế từ dữ liệu của chúng tôi. Công thức là:

$$\theta_{c,i} = \frac{\text{count}(w_i, c) + 1}{N + v} \tag{8}$$

	* $\text{count}(w_i, c)$ : số lần từ $w_i$ xuất hiện trong đoạn văn thuộc class $c$
	* $N$ : tổng số từ trong đoạn văn thuộc class
	* $v$ : kích thước của từ vựng (vocab size)
	
- Code Python:
```python
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    "text": [
        "Free money now",
        "Call now to claim your prize",
        "Meet me at the park",
        "Lets catch up later",
        "Win a new car today!",
        "Lunch plans?",
        "Congratulations! You won a lottery",
        "Can you send me the report?",
        "Exclusive offer for you",
        "Are you coming to the meeting?",
    ],
    "label": [
        "spam",
        "spam",
        "not spam",
        "not spam",
        "spam",
        "not spam",
        "spam",
        "not spam",
        "spam",
        "not spam",
    ],
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert labels to numeric
df["label"] = df["label"].map({"spam": 1, "not spam": 0})

# Features and labels
X = df["text"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

# Convert text to Bag of Words vectors
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Evaluate model
y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Test with a custom message
custom_message = ["Congratulations, you've won a free vacation"]
custom_vector = vectorizer.transform(custom_message)

prediction = model.predict(custom_vector)

print(
    "Prediction for custom message:",
    "Spam" if prediction[0] == 1 else "Not Spam",
)
```

---
### 2.3. Bernoulli Naive Bayes

- **Bernoulli Naïve Bayes** là một phân nhóm của Thuật toán Naïve Bayes. Thuật toán này thường được sử dụng khi dữ liệu ở dạng nhị phân và mô hình hóa sự xuất hiện của các đặc trưng bằng phân phối Bernoulli. Thuật toán này được sử dụng để phân loại các đặc trưng nhị phân như "Có" hoặc "Không", "1" hoặc "0", "Đúng" hoặc "Sai", v.v. Cần lưu ý rằng các đặc trưng này độc lập với nhau.

- Cốt lõi của Bernoulli Naïve Bayes dựa trên Định lý Bayes giúp tính toán xác suất có điều kiện của một lớp nhất định và đưa ra một số dữ liệu $\mathbf{x} = (x_1, x_2, \dots, x_N)$. Bây giờ trong mô hình Bernoulli Naïve Bayes, chúng tôi giả định rằng mỗi đặc trưng độc lập có điều kiện với lớp và. Điều này có nghĩa là chúng ta có thể tính xác suất xảy ra của từng đặc trưng như sau:

$$p(x_i \mid y) = p(i \mid y)x_i + (1 - p(i \mid y))(1 - x_i)$$

	* $p(x_i \mid y)$ : xác suất có điều kiện của $x_i$ xảy ra với điều kiện $y$ đã xảy ra.
	* $i$ : sự kiện
	* $x_i$ : giá trị nhị phân ($0 \text{ or } 1$)
- **Bernoulli Distribution**:
	- Phân phối Bernoulli được sử dụng để tính toán xác suất rời rạc. Nó tính toán thành công hoặc thất bại. Ở đây, biến ngẫu nhiên là $1$ hoặc $0$, với xác suất xảy ra được ký hiệu lần lượt là $p$ hoặc $(1 - p)$:

$$f(x) = \begin{cases} p^x(1 - p)^{1-x} & \text{if } x = 0, 1 \\ 0 & \text{otherwise} \end{cases}$$
## 3. Laplace Smoothing

> ⚠️ **Vấn đề Zero Probability:** Nếu một đặc trưng $x_i$ chưa từng xuất hiện cùng lớp $y$ trong tập Train, $P(x_i \mid y) = 0$. Khi nhân các xác suất lại, toàn bộ $P(y \mid X)$ sẽ bằng $0$, làm vô hiệu hóa các thông tin khác!

- Công thức Laplace Smoothing
	- Cộng thêm hằng số $\alpha > 0$ (thường chọn $\alpha = 1$) vào tử số và điều chỉnh mẫu số:

$$P(x_i \mid y) = \frac{\text{count}(x_i, y) + 1}{N_y + |V|}$$

	* $\text{count}(x_i, y)$: Số lần đặc trưng $x_i$ xuất hiện trong lớp $y$.
	* $N_y$: Tổng số lần tất cả các đặc trưng xuất hiện trong lớp $y$.
	* $\vert{}V\vert{}$: Kích thước từ vựng / tổng số đặc trưng độc lập.

---
## 4. Log Likelihood

> Vấn đề Numerical Underflow: Kích thước văn bản hoặc số lượng feature rất lớn. Dẫn đến xác xuất rất nhỏ xấp xỉ 0.

- Vì hàm $\ln$ hoặc $\log$ là môt hàm **đơn điệu tăng**, việc áp dụng $\log$ sẽ giữ nghiệm vị trí điểm cực đại:
$$\hat{y} = \arg\max_{y} \left[ \log P(y) + \sum_{i=1}^{n} \log P(x_i \mid y) \right]$$
## 5. So sánh các mô hình
| Aspect                          | Gaussian Naïve Bayes                                                         | Multinomial Naïve Bayes                                                     | Bernoulli Naïve Bayes                                                |
| :------------------------------ | :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------- |
| **Loại đặc trưng**              | Liên tục (các đặc trưng có giá trị thực)                                     | Rời rạc (dữ liệu đếm hoặc các đặc trưng dựa trên tần suất)                  | Nhị phân (đặc trưng nhận hai giá trị)                                |
| **Giả định**                    | Giả sử dữ liệu tuân theo phân phối chuẩn Gaussian                            | Giả sử dữ liệu tuân theo phân phối đa thức                                  | Giả sử dữ liệu tuân theo phân phối Bernoulli (nhị phân)              |
| **Trường hợp sử dụng phổ biến** | Phù hợp với các đặc trưng liên tục như chiều cao, cân nặng, v.v.             | Phù hợp để phân loại văn bản (số lượng từ)                                  | Phù hợp cho các tác vụ phân loại nhị phân (ví dụ: phát hiện thư rác) |
| **Biểu diễn dữ liệu**           | Các đặc trưng được coi là các biến liên tục                                  | Các đặc trưng được coi là số đếm hoặc tần số rời rạc                        | Các đặc trưng được coi là giá trị nhị phân (0 hoặc 1)                |
| **Mô hình toán học**            | Sử dụng phân phối Gaussian (trung bình và phương sai) cho mỗi đặc trưng      | Sử dụng phân phối đa thức cho số lượng từ trong phân loại văn bản           | Sử dụng phân phối Bernoulli (xác suất một đặc trưng hiện diện)       |
| **Ví dụ**                       | Dự đoán xem một email có phải là thư rác hay không dựa trên các đặc trưng số | Dự đoán liệu một tài liệu có phải là thư rác hay không dựa trên số lượng từ | Phân loại tài liệu là thư rác hay không dựa trên sự hiện diện của từ |