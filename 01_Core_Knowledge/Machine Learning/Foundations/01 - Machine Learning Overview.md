---
tags: [ml/basics, concept]
created: 2026-07-20
---

# What is Machine Learning (ML)?

## 1. Định nghĩa cốt lõi
Thay vì lập trình truyền thống (Traditional Programming) nơi con người viết ra **Quy tắc (Rules)** + nạp **Dữ liệu (Data)** để máy tính cho ra **Kết quả (Answers)**; thì **Machine Learning** đảo ngược quy trình đó:
> Chúng ta đưa **Dữ liệu (Data)** + **Kết quả (Answers)**, máy tính sẽ tự học để tìm ra **Quy tắc (Rules)**.
## 2. Định nghĩa kinh điển (Tom Mitchell - 1997)
Một chương trình máy tính được gọi là "học" từ **Kinh nghiệm $E$** (Experience) đối với một nhóm **Nhiệm vụ $T$** (Task) và **Thước đo hiệu suất $P$** (Performance), nếu hiệu suất của nó ở các nhiệm vụ trong $T$, được đo bởi $P$, cải thiện nhờ kinh nghiệm $E$.

*Ví dụ với bộ lọc Spam Email:* * **Task ($T$):** Phân loại email là Spam hay Non-Spam. * **Performance ($P$):** Tỷ lệ % email được phân loại chính xác. * **Experience ($E$):** Dữ liệu các email cũ đã được gán nhãn (Spam/Hợp lệ).

## 3. Bản chất của ML là gì?
Về mặt Toán học, học máy thực chất là bài toán **Xấp xỉ hàm số (Function Approximation)**. Giả sử có một hàm số thực tế $f(x) = y$ ẩn số mà ta không biết, ML cố gắng tìm ra một hàm $\hat{f}(x)$ sao cho $\hat{f}(x) \approx y$ với mọi dữ liệu đầu vào $x$.

## 4. Các chủ đề liên quan cần học tiếp

* Để biết ML có những hướng tiếp cận nào, xem: [[Types of Machine Learning]] 
* Để biết cách triển khai một dự án ML trong thực tế, xem: [[Machine Learning Workflow]] 
* Hai "cơn ác mộng" lớn nhất khi huấn luyện mô hình: [[Overfitting vs Underfitting]]