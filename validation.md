# 🔬 Validation Methodology

## Current Approach

| Item | Method |
|------|--------|
| Split | Random stratified 80/20 |
| Stratification | By AAMI class label |
| Reproducibility | Fixed `random_state=42` |
| Records | 10 records (100–109) |

**Known limitation:** Beats from the same patient/recording can appear in both
train and test sets. This may lead to optimistic performance estimates due to
within-patient signal correlation.

## Why This Matters

In clinical AI validation, **patient-wise (or record-wise) split** is the gold standard:

```
Random split (current):          Record-wise split (target):
┌─────────────────────┐          ┌─────────────────────┐
│ Record 100 beats:   │          │ Train records:       │
│  → 80% to train     │          │  100, 101, ... 230   │
│  → 20% to test      │          ├─────────────────────┤
│ (same patient!)     │          │ Test records:        │
└─────────────────────┘          │  231, 232, 233, 234  │
                                 └─────────────────────┘
```

## Planned Improvement

```python
# Target: record-wise holdout following AAMI EC57 recommendations
TRAIN_RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119","121",
    # ... up to record 230
]
TEST_RECORDS = ["231", "232", "233", "234"]  # Held-out test set
```

## Reproducibility

All experiments are fully reproducible:

```bash
# Exact reproduction of reported results
python train.py --use-real-data --data-dir ./mitbih_data \
  --records 100 101 102 103 104 105 106 107 108 109
# → Macro F1: 0.758  |  Weighted F1: 0.990
```

Fixed seeds: `numpy.random.seed(42)`, `sklearn random_state=42`
