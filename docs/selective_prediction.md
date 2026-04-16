# Selective Prediction Under Distribution Shift

Selective prediction adds a reject option to deployment: the model may abstain on examples it considers unreliable.

ShiftStat V4 focuses on a practical question:

- when does abstention genuinely improve accepted-set reliability under shift?

Core API:

- `shiftstat.selective.AbstentionPolicy`
- `shiftstat.selective.SelectivePredictor`
- `shiftstat.selective.evaluate_selective_under_shift(...)`

Supported baseline policies:

- confidence thresholding
- entropy-based certainty thresholding
- margin-based thresholding
- a simple learned-risk baseline using logistic regression on probability-derived uncertainty features

The library reports:

- retained coverage
- abstention rate
- selective accuracy and risk
- selective log loss and selective calibration
- subgroup-specific abstention summaries
