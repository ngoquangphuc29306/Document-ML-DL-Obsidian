---
title: "Feature Engineering trong Production"
tags: [feature-engineering, mlops, production]
---

# Feature Engineering trong Production

## 1. Train-serving skew

Xảy ra khi feature được tính khác nhau giữa train và inference.

Ví dụ:

```text
Train: missing → median
Production: missing → 0
```

hoặc timezone, window và business rule khác nhau.

## 2. Point-in-time correctness

Training row tại thời điểm \(t\) chỉ được dùng dữ liệu đã tồn tại trước \(t\).

Đây là yêu cầu bắt buộc với:

- Giao dịch
- Time series
- Customer history
- Fraud detection
- Recommendation

## 3. Feature contract

| Thuộc tính | Ví dụ |
|---|---|
| Name | `days_since_last_order` |
| Type | float |
| Definition | prediction time - last order |
| Source | orders |
| Freshness | 1 hour |
| Missing behavior | 999 |
| Version | v1 |

## 4. Batch và online feature

### Batch

- Tính theo giờ/ngày
- Phù hợp lịch sử dài hạn
- Độ trễ cao hơn

### Online

- Cần latency thấp
- Dùng cho request real-time
- Thường cần cache hoặc feature store

## 5. Monitoring

Theo dõi:

- Missing rate
- Mean/median
- Min/max
- Category mới
- Distribution drift
- Feature freshness
- Schema changes
- Out-of-range values

## 6. Data drift

$$
P_{\text{train}}(X) \neq P_{\text{production}}(X)
$$

Drift là tín hiệu cần điều tra, không tự động đồng nghĩa model đã hỏng.

## 7. Versioning

Khi thay công thức:

```text
customer_value_v1
customer_value_v2
```

Không nên âm thầm thay đổi feature cũ.

## 8. Reproducibility

Cần lưu:

- Dataset version
- Feature code version
- Time range
- Parameters
- Schema
- Model version

## 9. Inference pipeline

```text
Request
   ↓
Validate schema
   ↓
Fetch raw data
   ↓
Compute features
   ↓
Validate feature ranges
   ↓
Model prediction
   ↓
Log features + prediction
```

> [!TIP]
> Feature production tốt phải đúng, nhanh, ổn định, tái lập và giám sát được.
