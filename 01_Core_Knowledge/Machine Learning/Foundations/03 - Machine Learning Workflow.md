---
tags:
  - pipeline
  - ml/basics
created: 2026-07-20
---

# Machine Learning Workflow

Một dự án [[Machine Learning Overview]] thành công trong thực tế không chỉ có code thuật toán, mà nó tuân theo một quy trình chuẩn gồm các bước nghiêm ngặt sau:
![Pasted image 20260720235731](../../../05_Assets/Images/Pasted%20image%2020260720235731.png)
## 1. Các bước trong Workflow

### Bước 1: Data Collection (Thu thập dữ liệu)
*   Xác định bài toán cần giải quyết thuộc loại nào trong [[Types of Machine Learning]].
*   Thu thập dữ liệu từ Database, Web scraping, IoT Sensors, hoặc mua từ bên thứ ba. Dữ liệu là "xăng" để chạy cỗ máy ML.

### Bước 2: Data Preprocessing & Cleaning (Tiền xử lý dữ liệu)
*   Xử lý dữ liệu khuyết thiếu (Missing values), loại bỏ dữ liệu trùng lặp, xử lý các điểm dị biệt (Outliers).
*   Chuẩn hóa dữ liệu (Scaling, Normalization) để thuật toán chạy ổn định hơn.

### Bước 3: Feature Engineering (Trích xuất đặc trưng)
*   Biến đổi dữ liệu thô thành các tính năng (Features) giúp mô hình dễ học nhất.
*   Trong nhiều bài toán ML truyền thống, chất lượng dữ liệu và feature engineering có thể ảnh hưởng đến performance nhiều hơn việc chỉ thay đổi thuật toán.

### Bước 4: Data Splitting (Chia dữ liệu)
*   Chia tập dữ liệu thành 3 phần chính:
    *   **Train set (60-70%):** Dùng để huấn luyện mô hình.
    *   **Validation set (15-20%):** Dùng để tinh chỉnh siêu tham số (Hyperparameters).
    *   **Test set (15-20%):** Để riêng biệt, chỉ dùng để đánh giá khách quan sức mạnh cuối cùng của mô hình.

### Bước 5: Model Training (Huấn luyện mô hình)
*   Lựa chọn thuật toán phù hợp và đưa tập Train vào để máy tính tự tìm Quy tắc.
*   Đây là lúc cần theo dõi sát sao để tránh lỗi [[Overfitting vs Underfitting]].

### Bước 6: Model Evaluation (Đánh giá mô hình)
*   Dùng tập Test để tính toán các chỉ số đo lường (Metrics) như Accuracy, Precision, Recall, MSE, tùy thuộc vào bài toán.

### Bước 7: Deployment & Monitoring (Triển khai & Giám sát)
*   Đóng gói mô hình thành API và đưa lên server chạy thực tế.
*   Liên tục giám sát (Monitoring) vì dữ liệu thực tế luôn thay đổi theo thời gian (Data Drift).

---
> **Lời khuyên từ Chuyên gia:** ML Workflow không phải là một đường thẳng chạy một mạch từ bước 1 đến bước 7. Nó là một **Vòng lặp liên tục (Iterative Process)**. Đánh giá thấy mô hình tệ? Bạn sẽ phải quay lại bước Feature Engineering hoặc thu thập thêm dữ liệu!