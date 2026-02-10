# Complete Baselines Feature

## Summary

Implement Phase 3 from RECEIVER_GNN_FIX_PLAN.md: Complete the baseline models for receiver prediction.

## Status: COMPLETE

All requirements implemented and tested.

## Requirements

1. **Add frequency-weighted random baseline**: Random predictions weighted by training class frequency
2. **Add top-k accuracy for baselines**: Top-1, top-3, top-5 accuracy for all baselines
3. **Formatted baseline output table**: Present all baselines in a clear table format

## Implementation

### New functions in `models/receiver_model.py`:
- `frequency_weighted_baseline(train_labels, n_samples, seed=42)` - Random predictions weighted by class frequency
- `baseline_topk_accuracy(train_labels, test_labels, k=3)` - Top-k accuracy for frequency-based baselines
- `evaluate_all_baselines(train_labels, test_labels, test_node_features, seed=42)` - Comprehensive evaluation
- `print_baseline_table(baselines)` - Formatted output

### Updated `train_receiver()`:
- Replaced old baseline section with call to `evaluate_all_baselines()` and `print_baseline_table()`
- Stores full baseline metrics dict in results JSON

### Output format:
```
--- Baselines ---
                 Top-1   Top-3   Top-5
------------------------------------------
Majority class   0.XXX   0.XXX   0.XXX
Freq-weighted    0.XXX   0.XXX   0.XXX
Nearest player   0.XXX     N/A     N/A
Uniform random   0.043   0.130   0.217
```

## Test Cases (16 tests, all passing)

### TestFrequencyWeightedBaseline (5 tests)
- Returns correct shape
- Predictions within valid range (0-22)
- Respects class frequency distribution
- Deterministic with seed
- Different seeds produce different results

### TestBaselineTopkAccuracy (6 tests)
- Top-1 accuracy for majority class
- Top-3 includes more classes
- Top-5 accuracy
- Perfect accuracy when all in top-k
- Zero accuracy when none in top-k
- Handles 23 classes

### TestExistingBaselines (2 tests)
- Majority class baseline regression test
- Nearest player baseline regression test

### TestEvaluateAllBaselines (3 tests)
- Returns expected keys
- Uniform random has theoretical values
- Majority class top-1 is correct
