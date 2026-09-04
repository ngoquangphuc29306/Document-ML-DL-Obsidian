---
tags: [ml/evaluation, model-selection, validation]
created: 2026-07-21
---

# Cross-Validation

**Cross-Validation (CV)** là một kỹ thuật tái lấy mẫu (resampling technique) dùng để đánh giá hiệu suất của mô hình Machine Learning trên tập dữ liệu độc lập. Mục tiêu chính của Cross-Validation là đảm bảo mô hình có khả năng **khái quát hóa (Generalization)** tốt trên dữ liệu mới chưa từng thấy, thay vì chỉ học thuộc lòng tập Train.

---
## 1. Tại sao cần Cross-Validation?

Trong quy trình [[Machine Learning Workflow#Bước 4 Data Splitting Chia dữ liệu|ML Workflow]], phương pháp chia dữ liệu truyền thống là **Train/Test Split** (hoặc Train/Validation/Test Split):
* **Hạn chế của Train/Test Split thông thường:**
  1. **Nhạy cảm với sự phân chia ngẫu nhiên (High Variance of Evaluation):** Kết quả đánh giá mô hình phụ thuộc rất nhiều vào việc dữ liệu nào vô tình rơi vào tập Train, dữ liệu nào rơi vào tập Test.
  2. **Lãng phí dữ liệu (Data Waste):** Khi tập dữ liệu nhỏ, việc cắt ra 20-30% cho tập Test khiến mô hình bị thiếu dữ liệu huấn luyện.

> **Giải pháp của Cross-Validation:** Cho phép mọi điểm dữ liệu đều có cơ hội xuất hiện trong tập Huấn luyện (Train) lẫn tập Đánh giá (Validation) ở các lượt lặp khác nhau.

---

## 2. Các phương pháp Cross-Validation phổ biến

### 2.1. K-Fold Cross-Validation (Chuẩn mực phổ biến nhất)

#### Quy trình thực hiện:
1. Chia toàn bộ tập dữ liệu huấn luyện thành $K$ phần bằng nhau (gọi là $K$ **Folds**).
2. Thuật toán sẽ lặp lại $K$ lần (iterations/runs):
   * Tại mỗi lần lặp $i$, lấy **Fold thứ $i$** làm tập **Validation** (dùng để kiểm thử).
   * $K-1$ Folds còn lại kết hợp thành tập **Train** (dùng để huấn luyện mô hình).
1. Sau $K$ lần lặp, tính giá trị trung bình (Mean) và độ lệch chuẩn (Standard Deviation) của chỉ số đánh giá (như Accuracy, F1-score từ [[Evaluation Metrics]]) qua $K$ lần chạy.
![Pasted image 20260721102300](../../../05_Assets/Images/Pasted%20image%2020260721102300.png)
* **Giá trị $K$ thường chọn:** $K = 5$ hoặc $K = 10$ (giúp cân bằng tốt giữa thời gian tính toán và độ lệch [[Bias-Variance Tradeoff|Bias-Variance]]). 
--- 
### 2.2. Stratified K-Fold Cross-Validation (Dành cho Dữ liệu Mất Cân Bằng) 
* **Vấn đề của K-Fold thường:** Nếu tập dữ liệu bị mất cân bằng nhãn (Imbalanced Data - ví dụ: 95% nhãn 0 và 5% nhãn 1), việc chia ngẫu nhiên có thể dẫn đến một số Fold hoàn toàn không chứa nhãn 1 nào. 
* **Giải pháp:** **Stratified K-Fold** đảm bảo tỷ lệ phân bố giữa các nhãn (class distribution) trong mỗi Fold là **tương đương** với tỷ lệ của toàn bộ tập dữ liệu gốc. 
* **Ứng dụng:** Bắt buộc áp dụng cho bài toán [[Classification vs Regression#1 Classification|Classification]] khi dữ liệu bị lệch lớp nghiêm trọng. 
--- 
### 2.3. Leave-One-Out Cross-Validation (LOOCV) 
* **Bản chất:** Đây là trường hợp đặc biệt của $K$-Fold khi $K = N$ ($N$ là tổng số mẫu trong tập dữ liệu). 
* **Quy trình:** 
	* Mỗi lần lặp, mô hình huấn luyện trên $N-1$ mẫu và kiểm thử trên **đúng 1 mẫu** duy nhất. 
	* Lặp lại đúng $N$ lần. 
* **Ưu điểm:** Giảm tối đa hiện tượng Bias, tận dụng tối đa dữ liệu huấn luyện. 
* **Nhược điểm:** Tốn tài nguyên tính toán kinh khủng ($\mathcal{O}(N)$ lần train). Không áp dụng được cho tập dữ liệu lớn hoặc mô hình Deep Learning phức tạp. 
--- 
### 2.4. Time Series Split (Cross-Validation cho Dữ liệu Chuỗi Thời Gian) 
#### Tại Sao Không Dùng K-Fold Cross
-Validation Thông Thường?
Trong K-Fold Cross-Validation truyền thống, dữ liệu bị xáo trộn ngẫu nhiên (Random Shuffle). 

Nếu áp dụng cho Time Series:
* Dữ liệu ở tương lai ($t+1$) sẽ rơi vào tập **Train**, còn dữ liệu quá khứ ($t$) lại rơi vào tập **Validation**.
* Dẫn đến hiện tượng **Data Leakage (Rò rỉ dữ liệu tương lai)**, làm cho đánh giá mô hình bị lạc quan quá mức so với thực tế triển khai.

#### Walk-Forward Validation chia làm 2 loại:

![Pasted image 20260723222743](../../../05_Assets/Images/Pasted%20image%2020260723222743.png)
--- 