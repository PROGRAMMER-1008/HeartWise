# Data card

Source: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), Janosi, Steinbrunn, Pfisterer and Detrano (1989), DOI 10.24432/C52P4X. UCI declares CC BY 4.0. Original source records are cached separately in `data/reference`; application startup never downloads data. Retrieved 12 October 2026. The provenance of the local redistribution before this project is unknown; attribution to UCI does not establish its full redistribution history.

Profile `heartwise_redistributed_v1` requires SHA-256 `e31e52eb5ee890c1a11a3baef3df95e8174718d5b94b0a8b77665ec3c07328ff`. An arbitrary future CSV cannot silently inherit its target inversion. `src.audit.canonical_uci` is a distinct source importer using `num > 0`.

The reproducible full-field numeric audit (`python -m src.audit`) finds 303 unambiguous local matches to 302 distinct source rows; zero unmatched or ambiguous rows. One source record is duplicated and source row 184 (zero-based) is absent. No absent patient's values are invented. Exact mappings, hashes, source IDs and counts are in `data/mapping_manifest.json`.

| Field | Local meaning |
|---|---|
| cp 0 / 1 / 2 / 3 | Asymptomatic / atypical angina / non-anginal pain / typical angina |
| restecg 0 / 1 / 2 | Left ventricular hypertrophy / normal / ST-T wave abnormality |
| slope 0 / 1 / 2 | Downsloping / flat / upsloping |
| thal 1 / 2 / 3 | Fixed defect / normal / reversible defect (thallium test) |
| sex 0 / 1 | Female / male, as recorded by source |
| fbs 0 / 1 | Fasting blood sugar <=120 / >120 mg/dL |
| exang 0 / 1 | No / yes exercise-induced angina |
| ca 0–3 | Number of major vessels colored by fluoroscopy |
| target | Disease-present = 1 - local target = int(UCI num > 0) |

Age is in years, resting pressure in mmHg, cholesterol in mg/dL, maximum heart rate in beats/min; oldpeak is source ST depression induced by exercise relative to rest. These are not lifestyle or future-risk features. Source variables do not justify adding BMI/smoking/family-history inputs to this model.

Local `ca=4` and `thal=0` correspond to original missing `?`. Complete-case removal excludes seven local rows representing six distinct source patients. It leaves 296 rows and no exact duplicates. This is an explicit policy, not a claim that deletion is statistically optimal. Any imputation comparison must be a new training-fold-only study, never full-data imputation or final-test optimization.

Clinical/referral selection, historical measurement practices, limited demographic coverage, small sample size and redistribution anomalies restrict applicability. The observed cohort age range is 29–77. The service abstains outside it; this is a conservative dataset-domain policy, not a clinical age threshold. Training-partition ranges drive other warnings. Within-range values do not prove multivariate support.

Additional original hospital files were audited without training or tuning on them. Hungarian (294 rows), Switzerland (123), and VA Long Beach (200) have only 1, 0, and 1 complete compatible records respectively. Severe ca/thal and other missingness blocks a defensible external evaluation of this complete-input model. See `docs/external_cohort_audit.json`; no external accuracy or fairness claim is made.

A supplemental development-only diagnostic now compares complete-case training with fold-local median/mode imputation (`python -m scripts.imputation_study`). It includes the six distinct excluded missing-data records alongside the 236 development rows, never the 60 final-test records. Both methods are evaluated on the same 236 complete validation records across three folds. Brier scores are 0.11727 (complete-case) and 0.11739 (imputation); these are near-identical diagnostic results, not independent performance estimates or grounds to change policy. The study is versioned under `artifacts/studies`, with fold IDs and saved predictions. It does not establish validity for scoring incomplete specialist measurements; the deployed service still requires complete inputs.
