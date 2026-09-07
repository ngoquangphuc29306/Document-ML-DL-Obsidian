---
tags: [optimization, multi-objective-optimization, evolutionary-algorithms, genetic-algorithm, pareto, nsga-ii, ai]
aliases: [Multi-Objective Evolutionary Algorithms, MOEA, Tối ưu hóa đa mục tiêu, Pareto Optimization]
date: 2026-09-07
---

# Multi-Objective Evolutionary Algorithms (MOEA)

## 1. Bản chất của Tối ưu hóa Đa mục tiêu (Multi-Objective Optimization)

### 1.1. Khái niệm và Bài toán Đánh đổi (Trade-off)
* Trong thực tế, các vấn đề tối ưu thường yêu cầu thỏa mãn đồng thời từ hai mục tiêu trở lên ($m \ge 2$), trong đó các mục tiêu thường có mức độ ưu tiên ngang nhau [cite: 14].
* **Xung đột mục tiêu (Conflicting Objectives):** Việc cải thiện mục tiêu này thường dẫn đến sự suy giảm của mục tiêu khác [cite: 14].
* **Giải pháp lý tưởng (Ideal / Utopian Solution):** Là điểm tối ưu tuyệt đối trên mọi hàm mục tiêu [cite: 14]. Tuy nhiên, khi các mục tiêu xung đột, **giải pháp lý tưởng không bao giờ tồn tại trong thực tế** [cite: 14].
* **Mục tiêu của bài toán:** Thay vì tìm một nghiệm duy nhất như tối ưu đơn mục tiêu, mục tiêu ở đây là tìm ra một **tập hợp các giải pháp đánh đổi tối ưu (Optimal Trade-off Solutions)** [cite: 14].

![](../../../05_Assets/Images/Pasted%20image%2020260907110942.png)

---

### 1.2. Các Bài toán Thực tế Điển hình

#### Bài toán Chọn mua Điện thoại
* **Hai mục tiêu đối lập:** Giá thành thấp nhất ($f_1 \rightarrow \min$) và Hiệu năng cao nhất ($f_2 \rightarrow \max$) [cite: 14].
* **Đánh đổi thực tế:** Các dòng máy có cấu hình cao (Apple, Samsung cao cấp) luôn đi kèm mức giá đắt đỏ, trong khi các dòng giá rẻ (Xiaomi, Bphone, Oppo) phải chấp nhận hiệu năng hạn chế hơn [cite: 14]. Người dùng cần chọn một giải pháp tối ưu nằm trên đường vòng cung biên; các mẫu máy nằm thụt hẳn vào phía dưới đường vòng cung là phương án kém cỏi vì cùng tầm giá đó hoàn toàn có máy mạnh hơn [cite: 14].
![](../../../05_Assets/Images/Pasted%20image%2020260907105733.png)

#### Bài toán Tìm kiếm Kiến trúc Mạng Nơ-ron (Neural Architecture Search - NAS)
* **Ba mục tiêu đối lập:** Độ chính xác phân loại cao nhất (ImageNet Top-1 Accuracy $\rightarrow \max$), Số lượng tham số ít nhất (Params $\rightarrow \min$), và Khối lượng phép tính ít nhất (MAdds / FLOPs $\rightarrow \min$) [cite: 14].
* **Đánh đổi thực tế:** Các kiến trúc như EfficientNet-B1 đạt độ chính xác rất cao nhưng cồng kềnh; ngược lại các kiến trúc MobileNetV2, MobileNetV3 small tối ưu dung lượng nhỏ nhẹ cho thiết bị nhúng nhưng giảm độ chính xác [cite: 14]. Nhóm các mạng MUXNet, MixNet, MobileNet tạo nên dải đánh đổi cân bằng giữa tài nguyên phần cứng và hiệu quả [cite: 14].

![](../../../05_Assets/Images/Pasted%20image%2020260907105753.png)

#### Lập kế hoạch Xạ trị Áp sát Ung thư Tuyến tiền liệt (Brachytherapy)
* **Hai mục tiêu y khoa đối lập:** 
  1. Tối đa hóa liều bức xạ bao phủ khối u (Tumor coverage $\rightarrow \max$) [cite: 14].
  2. Tối đa hóa mức độ bảo vệ mô và cơ quan lành xung quanh như trực tràng, niệu đạo, bàng quang (Healthy organ sparing $\rightarrow \max$) [cite: 14].
* **Đặc tính kỹ thuật:** Gồm $\pm 16$ ống thông (catheters) với $\pm 200$ vị trí dừng của nguồn bức xạ (ứng với 200 biến quyết định biểu thị thời gian dừng) [cite: 14]. Không gian tìm kiếm phi tuyến tính (non-linear), không trơn (non-smooth) và không lồi (non-convex), đòi hỏi thời gian xử lý nhanh [cite: 14].

![](../../../05_Assets/Images/Pasted%20image%2020260907105821.png)
![](../../../05_Assets/Images/Pasted%20image%2020260907105835.png)
![](../../../05_Assets/Images/Pasted%20image%2020260907105844.png)
#### Huấn luyện Mô hình Học máy (Machine Learning Regression)
* **Hai mục tiêu đối lập:** Cực tiểu hóa sai số dự đoán (MSE Loss $\rightarrow \min$) và Cực tiểu hóa độ phức tạp của trọng số để chống Overfitting (Regularization $\|w\|_2^2 \rightarrow \min$) [cite: 14].
![](../../../05_Assets/Images/Pasted%20image%2020260907105947.png)

---

## 2. Lý thuyết Thống trị Pareto (Pareto Dominance)

Giả sử cần cực tiểu hóa (minimize) đồng thời $m$ hàm mục tiêu $f(x) = (f_1(x), f_2(x), \dots, f_m(x))$ [cite: 14].

### 2.1. Quan hệ Thống trị Pareto ($x^1 \prec x^2$)
Phương án $x^1$ được gọi là **thống trị Pareto** phương án $x^2$ (ký hiệu $x^1 \prec x^2$) khi và chỉ khi thỏa mãn đồng thời hai điều kiện sau [cite: 14]:
1. $x^1$ không kém hơn $x^2$ ở bất kỳ mục tiêu nào:
   $$\forall i \in \{1, 2, \dots, m\}: f_i(x^1) \le f_i(x^2)$$ [cite: 14]
2. $x^1$ thực sự vượt trội hơn $x^2$ ở ít nhất một mục tiêu:
   $$\exists i \in \{1, 2, \dots, m\}: f_i(x^1) < f_i(x^2)$$ [cite: 14]

![](../../../05_Assets/Images/Pasted%20image%2020260907110053.png)

### 2.2. Nghiệm Tối ưu Pareto, Tập Pareto và Biên Pareto
* **Nghiệm tối ưu Pareto (Pareto Optimal):** Điểm $x^1$ là tối ưu Pareto nếu không bị bất kỳ giải pháp hợp lệ nào khác thống trị:
  $$\neg \exists x^2: x^2 \prec x^1$$ [cite: 14]
* **Tập Pareto (Pareto Set - $P_S$):** Tập hợp tất cả các nghiệm tối ưu Pareto trong không gian quyết định (Decision Space):
  $$P_S = \{x^1 \mid \neg \exists x^2: x^2 \prec x^1\}$$ [cite: 14]
* **Biên Pareto (Pareto Front - $P_F$):** Tập hợp các vector giá trị mục tiêu của $P_S$ trong không gian hàm mục tiêu (Objective Space):
  $$P_F = \{f(x) = (f_1(x), f_2(x), \dots, f_m(x)) \mid x \in P_S\}$$ [cite: 14]

![](../../../05_Assets/Images/Pasted%20image%2020260907110112.png)

### 2.3. Các Hình thái của Biên Pareto
Tùy thuộc bài toán, biên Pareto có thể mang hình thái:
* **Convex (Lồi):** Hướng cong lồi quay về phía nghiệm lý tưởng [cite: 14].
* **Non-convex (Lõm):** Hướng cong lõm quay về phía nghiệm lý tưởng [cite: 14].
* **Hỗn hợp hoặc Rời rạc (Discontinuous / Combined):** Gồm nhiều đoạn cong ngắt quãng [cite: 14].
![](../../../05_Assets/Images/Pasted%20image%2020260907110216.png)
![](../../../05_Assets/Images/Pasted%20image%2020260907110153.png)
![](../../../05_Assets/Images/Pasted%20image%2020260907110224.png)

---

## 3. Các Phương pháp Chuyển đổi về Đơn mục tiêu & Giới hạn

### 3.1. Phương pháp Tổng có Trọng số (Weighted-Sum Approach)
Chuyển đổi bài toán đa mục tiêu thành một hàm mục tiêu tổng hợp duy nhất dạng tuyến tính [cite: 14]:

$$\min F(x) = \sum_{i=1}^{m} w_i f_i(x) \quad \left(\text{với } \sum_{i=1}^{m} w_i = 1 \text{ và } w_i \ge 0\right)$$

#### Cơ chế Thử các Bộ trọng số (Weight Sweeping)
* Mỗi lần chạy thuật toán tối ưu với một bộ trọng số $(w_1, w_2)$ cố định, ta chỉ tìm được **duy nhất 1 điểm tối ưu** trên biên [cite: 14].
* Để phác họa được đường biên Pareto, người ta phải chạy lại thuật toán rất nhiều lần với các tỷ lệ trọng số thay đổi dần đều [cite: 14]:
  * $(0.0 f_1 + 1.0 f_2)$: Tìm điểm cực trị trên trục $f_2$ [cite: 14].
  * $(0.4 f_1 + 0.6 f_2)$: Tìm điểm tối ưu nghiêng về $f_2$ [cite: 14].
  * $(0.5 f_1 + 0.5 f_2)$: Tìm điểm cân bằng ở trung tâm [cite: 14].
  * $(0.6 f_1 + 0.4 f_2)$: Tìm điểm tối ưu nghiêng về $f_1$ [cite: 14].
  * $(1.0 f_1 + 0.0 f_2)$: Tìm điểm cực trị trên trục $f_1$ [cite: 14].

> [!WARNING] Thất bại tuyệt đối khi Biên Pareto là Non-convex
> Về mặt hình học, bộ trọng số quy định độ dốc của các đường thẳng đồng mức (contour lines) [cite: 14]. Khi biên Pareto có dạng lõm (non-convex), đường thẳng đồng mức sẽ luôn tiếp xúc đầu tiên tại **hai điểm mút biên ngoài cùng** [cite: 14]. Do đó, dù ta có quét vô số bộ trọng số từ $0$ đến $1$, thuật toán vẫn chỉ tìm ra 2 điểm đầu mút mà hoàn toàn không thể tìm thấy bất kỳ điểm tối ưu nào ở vùng lõm giữa [cite: 14]!

![](../../../05_Assets/Images/Pasted%20image%2020260907110336.png)

![](../../../05_Assets/Images/Pasted%20image%2020260907110345.png)

![](../../../05_Assets/Images/Pasted%20image%2020260907110357.png)

---

### 3.2. Phương pháp Ràng buộc Epsilon ($\varepsilon$-Constraint Approach)
* **Cơ chế:** Chọn 1 hàm làm mục tiêu chính để tối ưu, đồng thời chuyển toàn bộ các hàm mục tiêu còn lại thành các điều kiện ràng buộc không vượt quá ngưỡng $\varepsilon$ [cite: 14]:
  $$\min f_k(x) \quad \text{với điều kiện } f_j(x) \le \varepsilon_j \quad (\forall j \ne k)$$
* **Ưu điểm:** Có khả năng tìm được nghiệm trên cả biên Convex lẫn Non-convex [cite: 14].
* **Nhược điểm:** Mỗi lần chạy vẫn chỉ tìm được 1 điểm; rất khó lựa chọn dải giá trị $\varepsilon$ để các điểm nghiệm phân bố trải đều đặn trên biên Pareto [cite: 14].
![](../../../05_Assets/Images/Pasted%20image%2020260907110423.png)

![](../../../05_Assets/Images/Pasted%20image%2020260907110430.png)
---

## 4. Cơ chế Xếp hạng và Chọn lọc trong MOEA

Để quần thể tiến hóa đa mục tiêu hiệu quả, giải thuật cần một quy tắc so sánh toàn diện giữa hai cá thể bất kỳ [cite: 14].

### 4.1. Domination Count (Số lần bị thống trị)
Với mỗi cá thể trong quần thể, đếm tổng số cá thể khác đang thống trị nó [cite: 14].
![](../../../05_Assets/Images/Pasted%20image%2020260907110448.png)
### 4.2. Xếp hạng Không bị thống trị (Nondominated Sorting)
Phân chia quần thể thành các tầng bậc (**Ranks**) [cite: 14]:
* **Rank 0 (Front $\mathcal{F}_0$):** Tập hợp các cá thể có $\text{Domination Count} = 0$ (không bị bất kỳ ai thống trị) [cite: 14].
* **Rank $i$ ($i > 0$):** Sau khi loại bỏ toàn bộ các cá thể ở các Rank nhỏ hơn ($0, 1, \dots, i-1$), các cá thể tiếp theo có $\text{Domination Count} = 0$ sẽ thuộc về Rank $i$ [cite: 14].
* *Nguyên tắc:* Cá thể ở Rank thấp hơn thì có chất lượng tốt hơn [cite: 14].
![](../../../05_Assets/Images/Pasted%20image%2020260907110515.png)

---

### 4.3. Khoảng cách Tập trung (Crowding Distance - CD)
* **Mục đích:** Đo lường mật độ phân bố xung quanh một cá thể để duy trì độ đa dạng (Diversity), ngăn ngừa quần thể dồn cục vào một góc [cite: 14].
* **Ý nghĩa:** Cá thể có CD lớn hơn nằm ở vùng thưa dân hơn, giúp mở rộng không gian tìm kiếm nên được ưu tiên chọn hơn [cite: 14].

#### Thuật toán Tính toán Crowding Distance trên cùng một Rank $P^r$
1. Đặt khoảng cách ban đầu cho mọi cá thể: $CD[x] \leftarrow 0$ [cite: 14].
2. Với mỗi hàm mục tiêu $f_k$ ($k = 1, \dots, m$):
   * Sắp xếp danh sách cá thể tăng dần theo giá trị $f_k$ [cite: 14].
   * Gán giá trị vô cùng cho hai cá thể ở biên cực trị: $CD[\text{đầu}] \leftarrow \infty$, $CD[\text{cuối}] \leftarrow \infty$ [cite: 14].
   * Với các cá thể nội suy ở giữa ($i = 2 \dots n-1$), cộng dồn khoảng cách chuẩn hóa [cite: 14]:
     $$CD[i] \leftarrow CD[i] + \frac{f_k(i+1) - f_k(i-1)}{f_k^{\max} - f_k^{\min}}$$

![](../../../05_Assets/Images/Pasted%20image%2020260907110619.png)

### 4.4. Toán tử So sánh Nghiệm (Crowded Comparison Operator)
Để so sánh hai cá thể $x$ và $y$ [cite: 14]:
1. Nếu $\text{Rank}(x) < \text{Rank}(y) \Rightarrow$ Chọn $x$ [cite: 14].
2. Nếu $\text{Rank}(x) > \text{Rank}(y) \Rightarrow$ Chọn $y$ [cite: 14].
3. Nếu $\text{Rank}(x) = \text{Rank}(y)$:
   * Nếu $CD(x) > CD(y) \Rightarrow$ Chọn $x$ (ưu tiên nghiệm ở vùng thưa) [cite: 14].
   * Nếu $CD(x) < CD(y) \Rightarrow$ Chọn $y$ [cite: 14].
   * Nếu $CD(x) = CD(y) \Rightarrow$ Chọn ngẫu nhiên giữa $x$ và $y$ [cite: 14].

---

## 5. Thuật toán Tiêu biểu: NSGA-II

NSGA-II (Deb et al., 2002) là thuật toán tiến hóa đa mục tiêu nền tảng sử dụng mô hình kế thừa tinh hoa **POPOP** [cite: 14]:

![](../../../05_Assets/Images/Pasted%20image%2020260907110651.png)


---

## 6. Tiêu chí Đánh giá Tập Xấp xỉ (Approximation Set Metrics)

Một tập xấp xỉ $S$ đạt chuẩn phải thỏa mãn 2 yếu tố: **Độ gần (Proximity)** tới biên thật $P_F$ và **Độ đa dạng (Diversity)** trải dài trên toàn bộ biên [cite: 14].

### 6.1. Front Occupation ($FO$) & Front Spread ($FS$)
* **Front Occupation ($FO$):** Số lượng giải pháp trong tập xấp xỉ: $FO(S) = |S|$ [cite: 14].
* **Front Spread ($FS$):** Đo lường độ dài sải rộng bao phủ của biên [cite: 14]:
  $$FS(S) = \sqrt{\sum_{i=1}^{m} \max_{(x^1, x^2) \in S \times S} (f_i(x^1) - f_i(x^2))^2}$$ [cite: 14]

---

### 6.2. Phân tích Chuyên sâu: Khoảng cách $D_{S \to P_F}$ và $D_{P_F \to S}$

#### Khoảng cách $D_{S \to P_F}$
* **Công thức toán học:**
  $$D_{S \rightarrow P_F} = \frac{1}{|S|} \sum_{x^1 \in S} \min_{x^2 \in P_S} \{d(x^1, x^2)\}$$ [cite: 14]
* **Ý nghĩa:** Đo khoảng cách trung bình từ từng điểm trong tập tìm được $S$ đến điểm gần nhất trên biên thật $P_F$ [cite: 14].
* **Nhược điểm chí mạng:** Chỉ đo được độ gần (Proximity) mà **hoàn toàn mù tịt về độ đa dạng (Diversity)** [cite: 14].
  * *Ví dụ:* Nếu thuật toán tìm ra 5 nghiệm nằm dồn sát rạt nhau thành một cụm siêu nhỏ, nhưng cụm này lại nằm trùng khít trên biên $P_F$, thì $D_{S \to P_F} = 0$ (điểm số tuyệt đối) [cite: 14]! Mặc dù $D_{S \to P_F} = 0$, tập nghiệm này lại cực kỳ tồi tệ do bỏ sót toàn bộ các vùng đánh đổi còn lại trên biên [cite: 14].
![](../../../05_Assets/Images/Pasted%20image%2020260907110726.png)

#### Vì sao phát minh thêm $D_{P_F \to S}$ để khắc phục?
* **Công thức toán học:**
  $$D_{P_F \rightarrow S} = \frac{1}{|P_S|} \sum_{x^1 \in P_S} \min_{x^2 \in S} \{d(x^1, x^2)\}$$ [cite: 14]
* **Cơ chế đảo góc nhìn:** Đứng từ **toàn bộ các điểm trên biên chuẩn $P_F$ rà ngược lại xem xung quanh nó có nghiệm nào của $S$ hay không** [cite: 14].
* **Hiệu quả:** Nếu $S$ bị dồn cụm ở một góc, các điểm ở những vùng còn lại của $P_F$ khi đo khoảng cách tới $S$ sẽ cho khoảng cách cực lớn, khiến giá trị $D_{P_F \to S}$ bị phạt tăng vọt [cite: 14]. Do đó, $D_{P_F \to S}$ chỉ nhỏ khi tập $S$ vừa gần biên thật, vừa phải **trải dài đồng đều khắp mọi ngóc ngách của biên** [cite: 14].

![](../../../05_Assets/Images/Pasted%20image%2020260907110743.png)

![](../../../05_Assets/Images/Pasted%20image%2020260907110752.png)
---

## 7. Các Kỹ thuật Nâng cao trong MOEA Hiện đại

### 7.1. Elitist Archive & Lưới Hypercubes
* **Vấn đề:** Quần thể $P$ có kích thước giới hạn, không thể lưu hết số lượng nghiệm non-dominated sinh ra liên tục [cite: 14].
* **Giải pháp:** Sử dụng thêm một quần thể phụ gọi là **Elitist Archive** để ghi nhớ mọi nghiệm non-dominated trong quá trình chạy [cite: 14].
* **Chia lưới Hypercubes (khi Archive đầy):**
  * Chia không gian mục tiêu thành lưới các siêu khối (hypercubes) [cite: 14].
  * Mỗi hypercube chỉ được chứa đúng 1 giải pháp [cite: 14]. Giải pháp mới chỉ được nhận nếu rơi vào ô trống hoặc thống trị được giải pháp cũ trong ô đó; ngăn chặn tình trạng nghiệm tập trung quá dày đặc [cite: 14].

![](../../../05_Assets/Images/Pasted%20image%2020260907110857.png)

### 7.2. Phân cụm (Clustering) & Niching
* Chia quần thể thành nhiều cụm (clusters) có quy mô bằng nhau trong không gian mục tiêu [cite: 14].
* **Niching:** Giới hạn việc ghép đôi và lai ghép chỉ diễn ra giữa các cá thể trong cùng một cụm [cite: 14]. Điều này tránh được nhược điểm của MOEA cổ điển: lai ghép hai cá thể ở hai đầu mút quá xa nhau thường tạo ra con cái nằm lọt thỏm vào vùng giữa không khả thi [cite: 14].
![](../../../05_Assets/Images/Pasted%20image%2020260907110914.png)
### 7.3. Học Mô hình Liên kết (Linkage Tree) & MO-GOMEA
* Xây dựng cấu trúc cây liên kết (Linkage Tree) để học sự phụ thuộc giữa các biến quyết định (đơn biến độc lập hay đa biến phụ thuộc) [cite: 14].
* Ứng dụng trong **MO-GOMEA** (Multi-Objective Gene-pool Optimal Mixing Evolutionary Algorithm) để thực hiện các phép tráo đổi gen theo khối tối ưu, chỉ chấp nhận nghiệm con nếu nó thống trị nghiệm cũ hoặc được nạp vào Elitist Archive [cite: 14].

---

## 8. Bảng Tổng kết các Thuật toán MOEA Phổ biến

| Thuật toán | Tác giả & Năm | Cơ chế cốt lõi |
| :--- | :--- | :--- |
| **NSGA-II** | Deb et al. (2002) | Non-dominated Sorting kết hợp Crowding Distance và mô hình POPOP [cite: 14]. |
| **MOEA/D** | Zhang et al. (2007) | Phân rã bài toán đa mục tiêu thành nhiều bài toán con đơn mục tiêu song song [cite: 14]. |
| **MAMALGAM** | Bosman et al. (2010) | Mô hình xác suất Gaussian thích ứng ước lượng phân phối nghiệm [cite: 14]. |
| **MO-GOMEA** | Luong et al. (2014), Bouter et al. (2017) | Kết hợp Elitist Archive, Clustering, Linkage Tree và Gene-pool Optimal Mixing [cite: 14]. |
