# 📊 Experiment Results

## Dataset

| Item | Detail |
|------|--------|
| Source | MIT-BIH Arrhythmia Database (PhysioNet) |
| Records used | 10 records: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109 |
| Total beats | 15,425 |
| Split method | Random stratified split (80% train / 20% test) |
| Class standard | AAMI EC57 — N / S / V |

## Class Distribution

| Class | Train | Test | Total |
|-------|-------|------|-------|
| N (Normal) | 11,761 | 2,940 | 14,701 |
| S (SVEB) | 33 | 9 | 42 |
| V (VEB) | 546 | 136 | 682 |

> ⚠️ Severe class imbalance — `class_weight='balanced'` applied in Random Forest.

## Model

- **Algorithm:** Random Forest (n_estimators=200, class_weight='balanced')
- **Features:** 14 hand-crafted features per beat (R-peak, QRS duration, P/T amplitude, ST level, statistical moments)
- **Input:** 1-second ECG segment (360 samples @ 360Hz) centred on R-peak

## Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| N (Normal) | 0.99 | 1.00 | **0.99** | 2,940 |
| S (SVEB) | 1.00 | 0.22 | 0.36 | 9 |
| V (VEB) | 0.91 | 0.92 | **0.92** | 136 |
| **Macro F1** | | | **0.758** | |
| **Weighted F1** | | | **0.990** | |

## Confusion Matrix

```
              Predicted
              N     S     V
Actual N   2928     0    12
       S      7     2     0
       V     11     0   125
```

## Key Observations

- **VEB (F1 0.92):** Clinically most critical class. The wide, bizarre QRS morphology and absent P-wave are well-captured by hand-crafted features.
- **SVEB (F1 0.36):** Only 42 samples across 10 records — insufficient for reliable generalisation. Expected to improve significantly with all 48 records.
- **N (F1 0.99):** Near-perfect detection of normal sinus rhythm.

## Feature Importance (Top 5)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | P amplitude | 0.153 |
| 2 | Kurtosis | 0.132 |
| 3 | T amplitude | 0.130 |
| 4 | Skewness | 0.099 |
| 5 | R amplitude | 0.098 |

> P-wave amplitude ranks #1 — clinically consistent, as VEB beats lack a P-wave entirely.
