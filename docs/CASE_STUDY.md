# Case study

HeartWise originally combined useful components—six classifiers, FastAPI, Streamlit, SHAP, calibration and PDF exports—but those components did not consistently describe the same model or dataset semantics.

The first repair was scientific rather than visual. A full-field comparison against the original UCI records reproduced 303 matches to 302 source rows. Chest-pain, ECG and slope labels were corrected without changing numeric encodings. Target inversion is now guarded by a named profile and exact hash. This replaces a correlation-based explanation of the target with reproducible source evidence.

The previous workflow selected calibration using final-test Brier scores and inferred constraints before cross-validation. The corrected procedure confines preprocessing/calibration selection to development folds, removes unsupported constraints, keeps logistic regression competitive, and uses nested development estimates. The final test is accessed only after locking the procedure. The already-inspected dataset still supports only internal retrospective evaluation.

The original confusion image implied TN 28, FP 4, FN 6, TP 22 while final saved metrics implied a different matrix. New figures and metrics derive from one saved final prediction table. This is a consistency correction, not evidence that model accuracy improved. The new model misses seven of 28 positives; the report makes that denominator visible.

Explaining a separately fitted uncalibrated tree did not reconstruct the calibrated ensemble score. Dropping inactive one-hot terms also lost attribution. The new explanation targets the actual served probability through raw-field permutation Shapley estimates. It includes all fields, base value, target and scale, plus a numerical reconstruction check. This costs about 9 ms p95 locally in the measured synthetic workload; it remains optional and visibly degradable.

The product now treats an assessment as an immutable session snapshot. Edits become stale indicators rather than silently changing the report. During AppTest verification, a numeric int/float normalization difference was found to clear review confirmation on reruns; normalizing widget initialization fixed it. Cross-page navigation also preserves form values and the assessed PDF. Browser review led to wrapping review/explanation tables rather than hiding category text in narrow scrollable cells.

Deployment consumes validated bundles and does not retrain. Historical artifacts are preserved. Original additional UCI hospitals were audited, but only 1/0/1 complete compatible records remain, preventing meaningful external evaluation without a separately justified missing-data model.

Lesson: traceable semantics, honest evaluation, and coherent artifact identity matter more than adding algorithms or advertising a larger accuracy number.

Evidence-matched resume description: Built an explainable research classifier with FastAPI and Streamlit, source-verified data mappings, training-only model selection, versioned artifacts, final-probability explanation checks, session-bound PDF exports and tested local container serving.
