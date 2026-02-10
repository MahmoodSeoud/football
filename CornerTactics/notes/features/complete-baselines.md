# Complete Baselines Feature

## Summary

Implement Phase 3 from RECEIVER_GNN_FIX_PLAN.md: Complete the baseline models for receiver prediction.

## Requirements

1. **Add frequency-weighted random baseline**: Random predictions weighted by training class frequency
2. **Add top-k accuracy for baselines**: Top-1, top-3, top-5 accuracy for all baselines
3. **Formatted baseline output table**: Present all baselines in a clear table format

## Changes Needed

### New functions in `models/receiver_model.py`:
- `frequency_weighted_baseline(train_labels, n_samples, seed=42)`
- `baseline_topk_accuracy(train_labels, test_labels, k=3)`

### Update `train_receiver()` baseline section:
- Add frequency-weighted baseline evaluation
- Compute top-k accuracy for all baselines where applicable
- Print formatted table:
  ```
  --- Baselines ---
                      Top-1    Top-3    Top-5
  Majority class      0.XXX    0.XXX    0.XXX
  Freq-weighted       0.XXX    0.XXX    0.XXX
  Nearest player      0.XXX    N/A      N/A
  Uniform random      0.043    0.130    0.217
  ```

## Test Cases

1. `frequency_weighted_baseline` returns correct distribution based on train labels
2. `baseline_topk_accuracy` correctly computes top-k accuracy
3. Baselines are evaluated and printed in `train_receiver()`
