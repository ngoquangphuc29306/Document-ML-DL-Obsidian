---
tags: [ml/core-concepts, evaluation]
created: 2026-07-20
---

# Overfitting vs Underfitting

Đây là hai vấn đề kinh điển mà mọi AI Engineer đều phải đối mặt hằng ngày khi xây dựng mô hình trong bước `Model Training` của [[Machine Learning Workflow]].

## 1. Bản chất của hai hiện tượng

![[Pasted image 20260721095208.png]]

### Underfitting 
*   **Hiện tượng:** Mô hình quá đơn giản, không học được cấu trúc của dữ liệu nền tảng.
*   **Biểu hiện:** Gây ra lỗi lớn ([[Bias-Variance Tradeoff|High Bias]]) trên **cả tập huấn luyện (Train set)** lẫn **tập kiểm thử (Validation set)**.
*   **Ví dụ:** Dữ liệu thực tế hình parabol nhưng bạn lại cố dùng một đường thẳng để dự báo.

### Overfitting 
*   **Hiện tượng:** Mô hình quá phức tạp, nó học thuộc lòng luôn cả những "nhiễu" (noise) và chi tiết thừa trong tập Train.
*   **Biểu hiện:** Lỗi rất thấp trên tập Train ([[Bias-Variance Tradeoff|Low Bias]]) nhưng lỗi lại cực kỳ cao khi gặp dữ liệu mới ở tập Test ([[Bias-Variance Tradeoff|High Variance]]). Mô hình mất đi khả năng **Khái quát hóa (Generalization)**.
![[Pasted image 20260721095112.png|1000]]

---

## 2. Bảng so sánh & Cách khắc phục

| Thuộc tính             | Underfitting (High Bias)                                                                                    | Overfitting (High Variance)                                                                                                                                 |
| :--------------------- | :---------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Lỗi trên Tập Train** | Cao                                                                                                         | Cực kỳ thấp                                                                                                                                                 |
| **Lỗi trên Tập Val**   | Cao                                                                                                         | Cao                                                                                                                                                         |
| **Nguyên nhân**        | Mô hình quá đơn giản; Thiếu tính năng (Features).                                                           | Mô hình quá phức tạp; Dữ liệu quá ít; Quá nhiều nhiễu.                                                                                                      |
| **Cách khắc phục**     | 1. Tăng độ phức tạp mô hình.<br>2. Thêm nhiều tính năng mới.<br>3. Giảm bớt các ràng buộc (Regularization). | 1. Thu thập thêm dữ liệu.<br>2. Giảm bớt số lượng tính năng.<br>3. Sử dụng **Regularization** (L1, L2, Dropout).<br>4. Early Stopping (Dừng sớm khi Train). |

> **Mục tiêu tối thượng:** Tìm ra "Điểm ngọt ngào" (Sweet Spot) - nơi mô hình vừa đủ phức tạp để học dữ liệu nhưng không quá phức tạp để bị nhiễu. Điều này được gọi là **Bias-Variance Tradeoff** (Sự đánh đổi giữa Bias và Variance).