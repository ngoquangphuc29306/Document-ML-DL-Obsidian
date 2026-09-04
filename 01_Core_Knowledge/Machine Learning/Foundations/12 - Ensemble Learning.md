---
tags:
  - ml/core-concepts
  - ensemble-learning
date: 2026-07-23
---
# <font color = "red">Ensemble Learning</font>

## 1. Ensemble Learning là gì ?

- Ensemble Learning là một kỹ thuật kết hợp các môm hình học máy (**week learner**) lại với nhau để tạo thành một mô hình mạnh hơn có hiệu suất cao hơn.
- Ensemble methos thường đạt performance mạnh trên nhiều bài toán, đặc biệt là structured/tabular data
- Có hai cách tiếp cận :
	1. Homogenous: các mô hình có cùng một thuật toán.
	2. Heterogeneous: các week leaners dùng các thuật toán khác nhau (SVM, KNN, decision Tree...).
## 2. Ensemble Learning Techniques.

### A. Bagging (Bootstrap Aggregating)

- Loại mô hình sử dùng là **Homogeneous week learners**
- Từ Dataset gốc, chia thành nhiều tập con thông qua phương pháp lấy mẫu ngẫu nhiên có hoàn lại (Sampling with replacement). Các tập con gọi là **Bootstrap dataset**.
- Huấn luyện **song song** các mô hình base trên từng bootstrap dataset.
- Khi lấy mẫu thì trung bình mỗi mẫu bootstrap sẽ chưa khoảng 63.2% các điểm dữ liệu duy nhất từ tập gốc. Những điểm không được chọn được gọi là **Out-of-Bag (OOB)** và rất hữu ích để đánh giá mô hình sau này.
![Pasted image 20260723210554](../../../05_Assets/Images/Pasted%20image%2020260723210554.png)
### B. Boosting

- Là một kĩ thuật học máy thuộc **ensemble learning**, nơi ta kết hợp nhiều **weak leaner**, một cách tuần tự để hình thành một **strong learner**. Cách ghép nối này giúp giảm sai số dự đoán và cải thiện khả năng tổng quát hóa vì mỗi bước bổ sung đều được thiết kế để đưa tổ hợp tiến gần đến lời giải đúng.
- Loại mô hình sử dùng là **Homogeneous week learner**.
- Boosting học theo chuỗi mỗi **weak learner** đến sau tập trung khắc phục những sai sót mà tổ hợp hiện tại còn mắc. Kết quả cuối cùng không phải là lấy riêng dự đoán của mô hình cuối, mà là một **weighted aggregation** (tạm dịch: gộp có trọng số) của tất cả các **weak learner**, trong đó mô hình “giỏi” có tiếng nói nặng ký hơn trong quyết định chung.

<div align="center">
  <img src="Pasted image 20260723210633.png" width="500">
  <img src="Pasted image 20260724201932.png" width="500">
</div>
### C. Stacking
![Pasted image 20260723210658](../../../05_Assets/Images/Pasted%20image%2020260723210658.png)