---
title: "Preprocessing Checklist"
tags: [preprocessing, checklist]
---

# Preprocessing Checklist

## Data
- [ ] Target rõ
- [ ] Prediction time rõ
- [ ] Schema/unit đúng
- [ ] ID columns được nhận diện
- [ ] Missing semantics được hiểu

## Split
- [ ] Split trước khi fit transformer
- [ ] Stratify nếu phù hợp
- [ ] Group split nếu có entity
- [ ] Time split nếu forecasting

## Numerical
- [ ] Có cần scaling?
- [ ] Outlier là lỗi hay tín hiệu?
- [ ] Skewness có cần transform?
- [ ] Scaler fit trên train?

## Categorical
- [ ] Nominal hay ordinal?
- [ ] Cardinality?
- [ ] Unknown category?
- [ ] Target encoding có OOF?

## Text
- [ ] Normalize hợp lý
- [ ] Vocabulary fit trên train
- [ ] BoW / TF-IDF / embedding?
- [ ] Duplicate / near-duplicate?

## Leakage
- [ ] Future information?
- [ ] Target-derived feature?
- [ ] Group contamination?
- [ ] Validation mô phỏng production?

## Pipeline
- [ ] Pipeline/ColumnTransformer
- [ ] Lưu preprocessing cùng model
- [ ] Inference dùng đúng transformer/version
