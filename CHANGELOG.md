# Changelog

## 1.2.0 (2026-09-30)

Corrections found by applying QSARCert to 180 real models (MLR, ridge and random forests on ESOL,
FreeSolv and Lipophilicity, random and scaffold splits; `validation/`).

### Fixed
- Applicability domain: membership is decided by leverage alone (h <= h*). Compounds with high
  leverage but small residuals were counted as inside the domain, and residuals (unknown for new
  compounds) took part in the decision. One influential outlier no longer fails the whole check;
  the status follows the coverage.
- Applicability domain: when h* >= 1 (e.g. fingerprints with more bits than compounds) the check is
  NOT_APPLICABLE instead of reporting meaningless leverages (they were clipped at 10). h* uses the
  rank of the design, which equals p + 1 for non-collinear descriptors.
- Applicability domain: residuals are standardised with the training residual standard deviation
  (n - p - 1 degrees of freedom); `assess_qsar_quality` now accepts `y_train_pred`. Before, a robust
  scale of the test residuals was used, which hid large test errors.
- Y-randomization: the OLS surrogate is NOT_APPLICABLE when its expected chance R^2, p/(n - 1), is
  0.5 or more. With fingerprints it reported a chance correlation for every model.
- Split leakage: raw descriptors were compared by cosine similarity, which is close to 1 for almost
  every pair of molecules; nearly every test compound was flagged as a duplicate. Fingerprints are
  now compared by Tanimoto similarity; for continuous descriptors only identical inputs are counted.
  Exact duplicates are identical vectors, not cosine >= 0.9999.
- Report provenance carried the version 1.0.0.

### Added
- `perform_y_randomization(..., estimator=...)`: refits the user's model on permuted responses and
  scores it by k-fold cross-validated R^2; also `expected_chance_r2` and a permutation p-value.
- `check_split_leakage(..., smiles_train=, smiles_test=)`: duplicate structures and shared
  Bemis-Murcko scaffolds (requires RDKit).
- `ApplicabilityDomainResult.design_rank`, `residual_scale`, `residual_scale_source`.
- Status `NOT_APPLICABLE`, which does not enter the overall decision.

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
