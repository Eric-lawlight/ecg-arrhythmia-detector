# ⚠️ Limitations & Honest Assessment

This document transparently describes the current limitations of this prototype,
following responsible ML practice.

## 1. Small Dataset (10 of 48 records)

Only 10 MIT-BIH records were used for initial training.

- SVEB class has only **42 samples total** — far below reliable learning threshold
- Record selection (100–109) may not represent the full diversity of arrhythmia patterns
- **Plan:** Extend to all 48 records; SVEB F1 expected to improve substantially

## 2. Random Split (Not Patient-wise)

The current 80/20 split is **random stratified**, not record-wise or patient-wise holdout.

- Risk: beats from the same recording appear in both train and test sets
- This can **inflate performance metrics** due to within-patient correlation
- **Plan:** Implement record-wise holdout (e.g., train on records 100–230, test on 231–234)
- Proper evaluation would follow AAMI EC57 recommended validation protocol

## 3. Hand-crafted Features Only

14 manually engineered features are used instead of raw waveform learning.

- Feature engineering introduces domain assumptions that may not generalise
- Cannot capture subtle morphological patterns beyond the defined features
- **Plan:** Add 1D CNN branch operating on raw 360-sample segments (PyTorch)

## 4. Single-lead Input

Only Lead II (channel 0) of the MIT-BIH signal is used.

- Clinical ECG typically uses 12 leads for comprehensive analysis
- Single-lead limits detection of certain arrhythmia subtypes
- Consistent with Fitbit Charge 5 hardware (Lead I equivalent)

## 5. No Temporal Context

Each beat is classified independently with no RR-interval or sequence modelling.

- Real arrhythmia patterns often depend on beat sequence (e.g., bigeminy)
- **Plan:** Add RR-interval features; explore LSTM for sequence-aware classification

## 6. Synthetic Demo Data

The built-in generator creates idealised ECG morphology for pipeline validation only.

- Synthetic beats are more separable than real data by design
- Macro F1 of 1.000 on synthetic data should not be compared to real-data results

---

## Summary Table

| Limitation | Severity | Status | Plan |
|-----------|----------|--------|------|
| 10 records only | High | Current | Extend to 48 records |
| Random split | Medium | Current | Record-wise holdout |
| Hand-crafted features | Medium | Current | Add 1D CNN |
| Single lead | Low | By design | Acceptable for wearable use case |
| No temporal context | Medium | Current | RR-interval features |
