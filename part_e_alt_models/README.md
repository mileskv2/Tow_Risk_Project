# Part E: alternative models

`Part_E_Alternative_Models.ipynb` (same code as `alt_models.py`) fits other models on the
Part C data and train/test split, using Part C's MASE. Run `python ../part_c/part_c_analysis.py`
first, because the extra-cars comparison reads Part C's output table.

1. Poisson and Negative Binomial GLMs: count models with the same predictors as Part C model 3
2. Logistic regression: does a 150+ tow day citywide happen tomorrow?
3. Regression with ARMA(1,1) errors: citywide dynamic regression

The CSVs in this folder are the outputs.
