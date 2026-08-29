---
tags:
  - ml/basics
  - classification
  - clustering
  - regression
created: 2026-07-20
---
	
# Types of Machine Learning

Học máy thường được chia thành 4 nhóm chính dựa trên cách thức mà mô hình tiếp nhận dữ liệu và học hỏi.

## 1. Supervised Learning
*   **Đặc điểm:** Dữ liệu huấn luyện **phải có nhãn (Labeled Data)**. Tức là với mỗi đầu vào $x$, ta đã biết trước đầu ra đúng $y$.
*   **Mục tiêu:** Học hàm số ánh xạ từ $x \rightarrow y$.
*   **Hai bài toán chính:**
    *   **Regression (Hồi quy):** Đầu ra là một giá trị liên tục (Ví dụ: Dự báo giá nhà, dự báo nhiệt độ).
    *   **Classification (Phân loại):** Đầu ra là các nhãn rời rạc (Ví dụ: Nhận diện ảnh Chó/Mèo, phân lọc Email Spam).
![[Classification.excalidraw|1000]]
### 1.1 Regression
Đối với một bài toán hồi quy, dữ liệu huấn luyện $\mathcal{D}_n$ có dạng một tập hợp gồm n cặp:

$$\mathcal{D}_n = \{(x^{(1)}, y^{(1)}), \ldots, (x^{(n)}, y^{(n)})\}$$

- $x^{(i)}$:  biểu diễn một đầu vào, thường là môt vector $d$-chiều gồm giá trị thực hoặc rời rạc  
- $y^{(i)}$: giá trị thực cần dự đoán (*target value*)  

Mục tiêu: với một đầu vào mới $x^{(n+1)}$, dự đoán giá trị $y^{(n+1)}$.  
Hồi quy thuộc loại *học có giám sát* vì mỗi ví dụ huấn luyện đều có đầu ra mong muốn.
### 1.2 Classification

Một bài toán phân loại giống như hồi quy, ngoại trừ việc các giá trị mà $y^{(i)}$ có thể nhận **không có thứ tự** hoặc có thứ tự trong bài toán ordinal classification như **low/medium/high**.
- **Phân loại nhị phân (Binary / Two-class)**:  
  Khi $y^{(i)}$ (còn gọi là *class*) được rút ra từ một tập hợp gồm **hai giá trị khả dĩ**.  
  Ví dụ:  
  - Email: *Spam* hoặc *Không spam*  
  - Ảnh: *Mèo* hoặc *Chó*
- **Phân loại đa lớp (Multi-class)**:  
  Khi $y^{(i)}$ được rút ra từ một tập hợp gồm **nhiều giá trị khả dĩ**.  
  Ví dụ:  
  - Loại trái cây: *Táo*, *Cam*, *Chuối*, *Xoài*  
  - Chữ số viết tay: 0 → 9
---
## 2. Unsupervised Learning

*   **Đặc điểm:** Dữ liệu đầu vào **không có nhãn (Unlabeled Data)**. Máy tính phải tự tìm ra cấu trúc ẩn sâu trong dữ liệu.
### 2.1 Clustering

Cho các mẫu $x^{(1)}, \ldots, x^{(n)} \in \mathbb{R}^d$, mục tiêu là tìm một phép phân hoạch (hay *clustering*) của các mẫu để nhóm những mẫu **tương tự nhau** lại với nhau.
Có nhiều mục tiêu khác nhau, tùy thuộc vào cách định nghĩa **độ tương tự (similarity)** giữa các mẫu và tiêu chí (criterion) được sử dụng, ví dụ:
- Giảm thiểu khoảng cách trung bình giữa các phần tử trong cùng một cụm.
- Tối đa hóa khoảng cách trung bình giữa các phần tử thuộc các cụm khác nhau.
Một số phương pháp thực hiện *soft clustering*, trong đó một mẫu có thể được gán mức độ thành viên phân số ở nhiều cụm.  
Ví dụ: một mẫu có thể có 0.9 thành viên trong cụm A và 0.1 trong cụm B.
Phân cụm đôi khi được sử dụng như một bước trong **ước lượng mật độ** (*density estimation*), hoặc để tìm ra **cấu trúc hữu ích** hay các **đặc trưng ảnh hưởng** trong dữ liệu.
## 3. Semi-Supervised Learning (Học bán giám sát)
*   **Đặc điểm:** Sự kết hợp giữa hai loại trên. Thường gồm một lượng nhỏ dữ liệu có nhãn và một lượng cực lớn dữ liệu chưa có nhãn (do chi phí gán nhãn rất đắt).

## 4. Reinforcement Learning (Học tăng cường)
*   **Đặc điểm:** Không có dữ liệu tĩnh. Mô hình (gọi là **Agent**) sẽ tương tác với một **Environment** (Môi trường) thông qua các hành động (**Actions**). Nhận về phần thưởng (**Rewards**) hoặc hình phạt (**Penalties**).
*   **Mục tiêu:** Tối đa hóa tổng phần thưởng nhận được theo thời gian (Ứng dụng: AlphaGo, xe tự lái).

---
*Xem thêm cách áp dụng các loại hình này vào quy trình thực tế tại: [[Machine Learning Workflow]]*