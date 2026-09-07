---
tags: [ml/evaluation, metrics]
created: 2026-07-20
---

# Evaluation Metrics (Chỉ số đánh giá mô hình)

Khi hoàn thành bước `Model Training` trong [[Machine Learning Workflow]], ta bắt buộc phải dùng tập Test hoặc Validation để đo lường sức mạnh của mô hình xem nó có gặp hiện tượng [[Overfitting vs Underfitting]] hay không.

## 1. Đối với bài toán [[Classification vs Regression#1. Classification|Classification]]

Để hiểu rõ các chỉ số này, ta cần dựa trên **Confusion Matrix** (Ma trận nhầm lẫn):
* **TP (True Positive):** Thực tế Đúng, Model đoán Đúng.
* **TN (True Negative):** Thực tế Sai, Model đoán Sai.
* **FP (False Positive):** Thực tế Sai, nhưng Model đoán Đúng (Sai lầm loại I).
* **FN (False Negative):** Thực tế Đúng, nhưng Model đoán Sai (Sai lầm loại II).

![Pasted image 20260720234501](../../../05_Assets/Images/Pasted%20image%2020260720234501.png)
### Các chỉ số cơ bản:

### Accuracy (Độ chính xác tổng quan)
* **Công thức:** $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* **Ý nghĩa:** Trong tất cả các mẫu, mô hình đoán đúng bao nhiêu phần trăm.
* **Lưu ý:** Gây hiểu nhầm nghiêm trọng khi gặp **Dữ liệu mất cân bằng (Imbalanced data)**. Ví dụ: Tập dữ liệu có 99% ảnh lành tính, 1% ảnh ung thư. Mô hình "ngốc" đoán bừa tất cả là lành tính vẫn đạt Accuracy 99%, nhưng nó đã bỏ sót 100% bệnh nhân ung thư.

### Precision (Độ chính xác của lựa chọn)
* **Công thức:** $$\text{Precision} = \frac{TP}{TP + FP}$$
* **Ý nghĩa:** Trong tất cả các trường hợp model **dự đoán là Positive**, có bao nhiêu trường hợp thực tế đúng là Positive.
* **Khi nào cần cao?** Khi hậu quả của việc đoán nhầm một mẫu Negative thành Positive là quá lớn (Sai lầm FP nguy hiểm). 
    * *Ví dụ:* Hệ thống chặn email nhầm một email công việc quan trọng (Negative) thành Spam (Positive).

### Recall / Sensitivity (Độ phủ / Tỷ lệ tìm sót)
* **Công thức:** $$\text{Recall} = \frac{TP}{TP + FN}$$
* **Ý nghĩa:** Trong tất cả các trường hợp **thực tế là Positive**, model đã quét và tìm ra được bao nhiêu phần trăm.
* **Khi nào cần cao?** Khi hậu quả của việc **bỏ sót** một mẫu Positive thực sự là cực kỳ nghiêm trọng (Sai lầm FN nguy hiểm).
    * *Ví dụ:* Bỏ sót một bệnh nhân mắc bệnh hiểm nghèo, bỏ sót một giao dịch quẹt thẻ tín dụng bị hack. Thà bắt nhầm (FP cao tí cũng được) còn hơn bỏ sót (FN phải cực kỳ thấp $\rightarrow$ Recall tiệm cận 100%).

### F1-Score
* **Công thức:** $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **Ý nghĩa:** Giá trị trung bình điều hòa (Harmonic Mean) giữa Precision và Recall. F1-Score chỉ cao khi cả Precision và Recall đều tốt, giúp đánh giá mô hình một cách toàn diện trên tập dữ liệu mất cân bằng.

---
