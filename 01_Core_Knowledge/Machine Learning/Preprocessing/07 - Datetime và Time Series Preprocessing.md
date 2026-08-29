---
title: "Datetime và Time Series Preprocessing"
tags: [preprocessing, datetime, time-series]
---

# Datetime và Time Series Preprocessing

> [!note]
> File này tập trung **chuẩn hóa dữ liệu thời gian**. Lag, rolling và domain time features nên đặt ở Feature Engineering.

## 1. Parse Datetime

Các format khác nhau cần chuyển về một kiểu datetime rõ ràng.

## 2. Timezone

Phải xác định:
- UTC/local;
- timezone production;
- daylight saving nếu có.

Timezone sai có thể làm lệch hour feature, daily aggregation và event matching.

## 3. Sort và Duplicate Timestamp

- sort theo time;
- kiểm tra timestamp trùng;
- hiểu một timestamp có nhiều entity hợp lệ hay không.

## 4. Missing Timestamp

Ví dụ hourly:

```text
10:00
11:00
13:00
```

thiếu `12:00`.

Có thể:
- reindex;
- giữ missing;
- interpolate;
- forward-fill.

## 5. Resampling

Minute → hour/day cần chọn aggregation đúng:
- sum;
- mean;
- last;
- max.

## 6. Time-based Split

Không random split khi mục tiêu là forecasting.

```text
Past → Train
Later → Validation
Future → Test
```

> [!important]
> Vấn đề của K-Fold thông thường không chỉ là shuffle; chính là nó có thể làm training chứa quan sát xảy ra **sau** validation.
