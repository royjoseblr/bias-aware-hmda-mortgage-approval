# Methodology

## Population and sample
Use filtered 2023 Michigan HMDA applications with `action_taken` equal to 1 or 3. A stratified analytical sample may be drawn after data-quality checks, preserving outcome and demographic representation.

## Outcome coding
- `action_taken = 1` -> `outcome = 1` (loan originated)
- `action_taken = 3` -> `outcome = 0` (application denied)

## Pipeline
1. Validate schema and filter outcomes.
2. Remove duplicate records where justified and quantify missingness.
3. Exclude leakage and post-decision fields.
4. Split data before fitting preprocessing.
5. Impute, encode, and scale through scikit-learn pipelines.
6. Compare logistic regression, random forest, and XGBoost.
7. Evaluate accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
8. Test group-outcome association with chi-square tests.
9. Calculate group selection rates, statistical parity difference, and disparate impact ratio.
10. Explain the selected model with SHAP.
11. Apply controlled mitigation and report accuracy-fairness trade-offs.

## Validity controls
- fixed random seed
- stratified split
- cross-validation on the training partition
- untouched holdout test set
- metrics with group sample sizes
- explicit leakage review
- reproducible configuration
