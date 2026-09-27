# Changelog

## 1.1.0 (2026-09-27)

Corrections to the OECD Principle 4 metrics, checked against independent implementations of the
published formulas (Consonni et al. 2009; Golbraikh & Tropsha 2002; Roy et al. 2012; Lin 1989).

### Fixed
- Golbraikh–Tropsha and Roy r_m² now use r², the squared Pearson correlation between observed and
  predicted values, rather than the coefficient of determination 1 − SS_res/SS_tot. The two differ
  whenever predictions are biased.
- Q²_F3 used the evaluation-set variance and therefore duplicated Q²_F2. It now uses the training
  variance and is NaN when no training responses are given.
- Q²_F1 uses the training mean. `assess_qsar_quality` now passes `y_train` through; previously
  Q²_F1 was silently equal to Q²_F2.
- The acceptance rules now require r² > 0.6 (was 0.5) and |R0² − R'0²| < 0.3 (was missing).
  A failing r² is listed in the diagnostic.
- Y-randomization: cR²_p now compares the ridge surrogate on original and permuted responses,
  so both terms come from the same model. Before, it used the external test R² of the user's model.

### Added
- `OECDValidationResult.r2_pearson`, `OECDValidationResult.r0_diff`, and the `y_train` argument of
  `calculate_oecd_metrics`.
- Tests against textbook formulas.
