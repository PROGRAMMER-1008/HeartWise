# HeartWise: project review and Codex implementation plan

Review date: 12 October 2026. Scope: the supplied project folder, all seven Python source files, dataset, saved metrics and metadata, the confusion-matrix image, dependency declarations, and container configuration. This document is an implementation backlog, not a statement that the changes have already been made.

## 1. Overall assessment

HeartWise is already a substantial educational machine-learning project: it combines a Streamlit interface, FastAPI service, shared preprocessing, six candidate classifiers, five-fold evaluation, calibration, local SHAP factors, a performance dashboard, downloadable PDF reports, and Docker deployment. Preserve these strengths.

The greatest improvement would be to make every visible claim traceable to the exact data and model being served. Fix category semantics, evaluation leakage, explanation fidelity, and inconsistent reports before investing in a new frontend or more algorithms. An honest, reproducible project with excellent demonstrations will be more impressive than a larger project with unsupported metrics.

Recommended positioning: **an explainable educational demonstration of Cleveland-dataset disease-status classification**. The dataset does not establish future heart-attack risk, a time horizon, or suitability for general consumer screening. Several required inputs come from specialist testing, so the present form is better suited to research demonstrations than patient self-assessment.

## 2. What was verified, and what remains unverified

### Direct checks

- Parsed all seven Python source files successfully with Python's syntax parser.
- Audited `data/heart.csv`: 303 rows; raw target counts 165 for class 1 and 138 for class 0; no blank or question-mark cells; one exact duplicate row. The existing loader removes seven special-code rows, leaving 296 rows with no exact duplicates.
- Compared the local CSV to the original public UCI processed Cleveland data, matching on eight unchanged fields: age, sex, resting blood pressure, cholesterol, fasting-blood-sugar flag, maximum heart rate, exercise-angina flag, and oldpeak. Every local row matched exactly one source row. However, only 302 distinct source rows were represented: one source row is duplicated locally and another is absent. This is not a perfect one-to-one copy.
- All 303 matched local rows have `1 - local_target == int(original_UCI_num > 0)`. The existing target inversion is correct for this exact file; its explanation should cite this comparison instead of inferring truth from correlations.
- Verified source category mappings and special missing-value codes; see section 3.
- Inspected the saved confusion matrix: TN 28, FP 4, FN 6, TP 22. Its accuracy is 50/60 = 83.3%, while the final metadata reports 86.7%. The code generates this plot before calibration and final decisions.
- Tried a prediction in the available local Python environment. Model loading emitted version warnings, and inference failed with `AttributeError: 'ColumnTransformer' object has no attribute '_name_to_fitted_passthrough'`. Installed scikit-learn is 1.3.2; artifacts were saved with 1.8.0. ReportLab is also absent from this interpreter. These are observed environment limitations, not proof that the pinned Docker environment fails.
- No README, test suite, CI configuration, project packaging configuration, or AGENTS.md was found in the supplied folder. The folder is not currently recognized as a Git repository.

### Limits of this review

The full app was not run in a matching environment; browser layout, accessibility, actual PDF rendering, Docker builds, concurrent inference, and all dependency pins remain unverified. No retraining was performed and no original artifact was replaced. Recommendations about those areas distinguish code observations from checks future implementation must perform. The implementation must verify dependency availability rather than assuming any pin is invalid.

### Current saved results: historical, not a clean future benchmark

| Measure | Saved value |
| --- | --- |
| Model / calibration | XGBoost / isotonic |
| Version | 2.2.0 |
| Cleaned train / test rows | 236 / 60 |
| Test accuracy | 86.7% |
| Test recall | 82.1% |
| Test ROC-AUC | 0.9381 |
| Base-model CV ROC-AUC | 0.8812 ± 0.0277 |
| Final false negatives / false positives | 5 / 3 |

These values are useful historical evidence. They should not be advertised as unbiased validation of the final calibrated model because calibration selection uses this same test set. The CV values describe the base candidate, not the complete calibrated selection procedure. Do not require future fixes to preserve or increase these numbers.

## 3. Highest-priority finding: the displayed categories are wrong

`src/data_pipeline.py:69` defines the labels used by the form, reports, explanation labels, and API documentation. Source matching verified these local-code meanings:

| Feature | Local code | Meaning verified against UCI | Present label |
| --- | --- | --- | --- |
| cp | 0 | Asymptomatic | Typical angina |
| cp | 1 | Atypical angina | Atypical angina |
| cp | 2 | Non-anginal pain | Non-anginal pain |
| cp | 3 | Typical angina | Asymptomatic |
| restecg | 0 | Left ventricular hypertrophy category | Normal |
| restecg | 1 | Normal | ST-T wave abnormality |
| restecg | 2 | ST-T wave abnormality | Left ventricular hypertrophy |
| slope | 0 | Downsloping | Upsloping |
| slope | 1 | Flat | Flat |
| slope | 2 | Upsloping | Downsloping |

Matching counts: cp codes 0/1/2/3 occurred 143/50/87/23 times; restecg 0/1/2 occurred 147/152/4 times; slope 0/1/2 occurred 21/140/142 times. The existing nonmissing thal mapping is consistent with UCI: local 1/2/3 corresponds to original 6/3/7, meaning fixed defect/normal/reversible defect.

This can make someone select a clinical description and submit the opposite category to the trained model. It also makes explanation text misleading. Correcting only the label strings preserves encoded model inputs; migrating encodings instead requires an explicit conversion and retraining. Never mix these approaches silently.

Additional provenance finding: local `ca=4` maps to original missing `?`, as does local `thal=0`. The seven removed local rows correspond to six distinct missing-data source patients because one is duplicated. Keep this distinction in the data-quality notes. Rename the user-facing `Thalassemia` label to a source-appropriate thallium stress-test result label; the existing categories describe test defects, not a thalassemia diagnosis.

Source: [UCI dataset and variable documentation](https://archive.ics.uci.edu/dataset/45/heart+disease), [original processed Cleveland records](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data). The mappings above are findings from the row comparison performed during this review, not mappings copied from a different redistributed dataset.

Local dataset SHA-256 at review time: `e31e52eb5ee890c1a11a3baef3df95e8174718d5b94b0a8b77665ec3c07328ff`.

## 4. Prioritized implementation backlog

Priority: P0 = correctness and trust; P1 = reliability and polished core experience; P2 = project differentiation; P3 = optional expansion. Effort is relative, not a delivery promise: S = localized work; M = several connected changes; L = substantial workflow work. IDs are stable so they can be referenced in prompts.

| ID | Improvement | Priority | Effort | Depends on |
| --- | --- | --- | --- | --- |
| HW-01 | Verified data dictionary and provenance | P0 | M | None |
| HW-02 | Leakage-free model and calibration evaluation | P0 | L | HW-01 |
| HW-03 | Faithful, complete model explanations | P0 | L | HW-02 |
| HW-04 | Consistent metrics, plots, claims, and artifacts | P0 | M | HW-02 |
| HW-05 | Unified validation and input-domain warnings | P1 | M | HW-01 |
| HW-06 | Reliable API and explicit serving modes | P1 | M | HW-05 |
| HW-07 | Guided form and durable assessment state | P1 | M | HW-05, HW-06 |
| HW-08 | Accessible visual design and clear results | P1 | M | HW-03, HW-07 |
| HW-09 | Reliable, polished PDF exports | P1 | M | HW-03, HW-04, HW-07 |
| HW-10 | Reproducible environments and clean packaging | P1 | M | None |
| HW-11 | Meaningful tests and continuous checks | P1 | L | Incremental with HW-01–10 |
| HW-12 | Reproducible builds and deployment readiness | P1 | M | HW-04, HW-06, HW-10 |
| HW-13 | Privacy, security, and responsible product wording | P1 | M | HW-05, HW-06 |
| HW-14 | Rich evaluation and model-card dashboard | P2 | M | HW-02, HW-04 |
| HW-15 | Transparent threshold and what-if playground | P2 | M | HW-02, HW-03, HW-05 |
| HW-16 | External validation and subgroup analysis | P2 | L | HW-01, HW-02 |
| HW-17 | Performance, observability, and artifact lifecycle | P2 | M | HW-06, HW-12 |
| HW-18 | README, case study, and portfolio demonstration | P1/P2 | M | Core correctness tasks |
| HW-19 | Optional batch workflow and saved research sessions | P3 | L | HW-05, HW-13 |
| HW-20 | Optional frontend or infrastructure expansion | P3 | L | Measured need |

### HW-01 — Establish a trustworthy data contract

Evidence: `src/data_pipeline.py:69–85` contains incorrect cp/restecg/slope labels; line 147 flips the target; data-quality text in the app relies on inferred correlations and hardcoded row counts.

Implement:

- Correct the label mappings in section 3 while retaining the existing numeric encodings initially. Check API examples, defaults, explanations, plot labels, and PDF values together.
- Add `docs/DATA_CARD.md` and a machine-readable mapping manifest containing source URL/DOI, retrieval date, license, file hashes, original-to-local encodings, target semantics, missing-value policy, duplicate/missing-row findings, and limitations.
- Add an explicit dataset profile, e.g. `heartwise_redistributed_v1`, so `load_dataset()` does not invert arbitrary future input files. A canonical UCI import should separately use `num > 0`.
- Create a reproducible verification script that compares records with numeric normalization, reports ambiguous/unmatched/duplicate rows, and verifies all available fields. Cache any downloaded source under a clearly separated reference-data directory; keep application startup offline.
- Make the current drop policy explicit. Compare complete-case removal with train-fold-only imputation later; never impute before splitting. Do not “repair” an absent source patient by inventing values.
- Derive data-quality counts from the run manifest. Replace claims about a broadly broken public dataset with the exact, source-verified transformation for this file.

Done when: mapping regression fixtures pass; an audit reproduces the 303 local matches/302 distinct source matches; wrong dataset profiles fail clearly; all displayed category meanings agree with the data dictionary.

### HW-02 — Remove leakage and evaluate the complete decision procedure

Evidence: `src/train.py:421–435` derives supervised monotonic constraints from all training labels before CV; lines 536–563 choose calibration using held-out test Brier scores. Validation-fold labels therefore affect constraints, and the test set becomes a calibration-selection set.

Implement:

- Keep one final test partition inaccessible to model, hyperparameter, calibration-method, and threshold selection. Persist row identifiers/hashes and split assignments.
- Use training-only inner CV to compare complete procedures: preprocessing, classifier, constraint inference, calibration, and threshold policy. Fit all data-derived decisions inside their training fold. Use repeated outer CV or nested CV for development estimates where feasible.
- Set calibration ensemble behavior explicitly and document it. Compare no calibration and sigmoid; treat isotonic as an experimental candidate with a small-sample warning. The current 236-row training set yields roughly 47 calibration examples per fold, a weak basis for a flexible calibrator.
- Use a bounded, reproducible hyperparameter search rather than a costly unbounded search. Keep logistic regression as a serious baseline: it is interpretable and leads the present CV ranking.
- Make the monotonic policy explicit: either justified fixed constraints or a fold-local learned policy. A strong correlation alone does not prove monotonicity. Remove unsupported “binned check” claims unless a reproducible check exists.
- Predeclare the selection objective and tie-break rules. The existing weighted score is a project choice, not an established clinical utility measure. Do not assert that XGBoost recall is “at least as good” when 0.778 is below Random Forest's 0.787.
- After locking the complete procedure, evaluate it once on the final test set. If reusing this already-inspected dataset, label results as internal retrospective evaluation; a newly shuffled split does not create independent external evidence.

Done when: tests demonstrate fold isolation; selection code receives no test labels; base versus calibrated results are explicit; the final report includes counts, uncertainty, threshold, split identity, and a reproducible run manifest. Do not promise a target accuracy.

Reference: [scikit-learn 1.8 CalibratedClassifierCV](https://scikit-learn.org/1.8/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) documents ensemble averaging and warns about isotonic overfitting with small calibration sets.

### HW-03 — Explain the model that actually produced the score

Evidence: `src/explain.py:35–45` loads a separate base pipeline; line 56 scores the calibrated model; lines 110–116 remove inactive one-hot contributions. Calibration with the default ensemble can average several independently fitted classifiers, so a separately trained base model is not simply the same prediction rescaled.

Implement:

- Choose and document an explanation contract. Either explain the final calibrated prediction with a suitable method/background, or expose base-model explanations separately and label them clearly as not additive explanations of the final probability. Benchmark the cost before choosing a method.
- Aggregate **all** one-hot SHAP contributions belonging to each raw clinical field, including inactive categories. Display one field with its actual selected category. Dropping inactive contributions discards attribution mass.
- Return explanation target/model ID, output scale (e.g. log-odds versus probability), expected/base value, complete grouped contributions, and an optional top-five view with a residual contribution.
- Add an additive reconstruction check on the explained model's own scale. Never show raw-margin SHAP values as percentage-point changes in calibrated probability.
- Explanations describe model associations, not causes or recommended medical actions. Make unavailable explanations a visible supported state; do not silently imply they succeeded.
- Support the winning model family honestly. Logistic regression can have efficient explanations; it need not lose selection just to retain tree SHAP.

Done when: aggregation preserves the full sum within tolerance; selected-category labels are correct; explanation metadata identifies the actual target; failures are visible; no chart claims to reconstruct a different model's probability.

### HW-04 — Generate one internally consistent artifact bundle

Evidence: `src/train.py:531` plots uncalibrated predictions; line 582 later calculates final predictions. The current matrix is [[28,4],[6,22]], while the final error report implies [[29,3],[5,23]]. `reports/model_comparison.csv` describes base candidates but the UI mixes it with final calibrated metrics. The saved selection rationale also contains the inaccurate recall comparison.

Implement:

- Produce confusion matrix, final ROC/PR curves, calibration curve, error analysis, final metrics, and metadata from one saved final prediction table and one threshold.
- Keep candidate/base plots explicitly labeled and separate from the final served model. Show whether CV statistics apply to candidates or complete procedures.
- Store artifact/run ID, schema version, dataset hash, split hash, config, random seeds, training code revision when available, dependency versions, class order, threshold/band policy, and model checksums.
- Write a complete bundle to a staging directory and publish it only after validation. Avoid partially updated files and stale SHAP plots left behind when generation fails.
- Use schema validation for metadata and plot manifests. Have the dashboard show an intentional placeholder when a chart is absent.

Done when: recomputing metrics from saved predictions reproduces every reported count/value; all figures carry the correct model/run identity; invalid or incomplete bundles cannot be activated; explanatory text is generated from actual comparisons.

### HW-05 — Share strict validation and communicate unsupported inputs

Evidence: API validation uses Pydantic, while local validation checks `int(record[col])` for categories and assumes comparable numeric values. A fractional category can be truncated by the local helper; strings may crash its numeric checks. Broad accepted ages 1–120 contrast with observed ages 29–77.

Implement:

- Use one schema or validation service for API, local inference, demo scenarios, and any later batch imports. Reject fractional categories, booleans masquerading as integers, nonfinite values, missing fields, unsupported categories, and unexpected fields according to an explicit contract.
- Bound and normalize the optional report identifier; escape it for its rendering context. Keep identifiers out of model features.
- Separate hard input-format bounds from observed training support. Show warnings for out-of-training-range inputs; an in-range value is not a guarantee that the entire patient profile is supported.
- Do not present child or very unusual-age predictions as supported just because the widget accepts them. Document the adult research population; use a domain status or abstention policy with a stated basis.
- Add field help, measurement units, and an explicit unknown state. Initially block scoring until required specialist measurements are supplied rather than substituting invented “normal” values.

Done when: the same invalid record is rejected in both modes; boundary/nonfinite/type tests pass; warnings appear consistently in API, UI, and report; IDs cannot break markup or filenames.

### HW-06 — Make API behavior reliable and serving mode explicit

Evidence: `app/streamlit_app.py:212–225` falls back locally after any request exception, including API validation/server errors. Local metadata and figures remain the dashboard source even when a different remote model scores. Startup always warms the local model. `/health` returns HTTP 200 based on object presence even if warmup failed; exception text is returned to clients.

Implement:

- Add configured `local`, `api`, and optionally `auto_demo` modes. API mode must work without local model files. Reserve fallback for the documented demo mode and connection-unavailable conditions, not 4xx validation errors or arbitrary server failures.
- Read model information from the serving source; identify the same artifact in predictions, dashboard, and report. Invalidate caches when the artifact changes.
- Use FastAPI lifespan startup; validate artifacts and a canary inference. Separate liveness from readiness. Readiness must return a failure status when inference cannot run. Explanation degradation can be distinct from prediction readiness.
- Use safe client-facing errors, structured server logs, request IDs, configurable timeouts, and validated response schemas. Retry only safe transient failures with a small bound; avoid duplicate expensive work.
- Use constrained response types/ranges and expose threshold, domain warnings, explanation status, and artifact ID. Distinguish application/API version from model version.

Done when: remote-only mode starts without local artifacts; 422 responses remain validation errors; unavailable model readiness fails; remote results never display unrelated local metrics; users get recoverable errors rather than stack traces.

### HW-07 — Improve the form and preserve the assessment

Evidence: `app/streamlit_app.py:300` renders results only inside `if submitted`; the patient, result, and PDF are not retained as a coherent session snapshot. Ordinary reruns can remove the result. Many clinical inputs have prefilled values with no demo-versus-real distinction.

Implement:

- Start with a choice between clearly marked synthetic demo cases and entering a complete research record. Provide named examples without storing real patient identifiers.
- Group the form into understandable steps: basic information, laboratory/ECG data, and specialist/exercise results. Add short tooltips and a review step.
- Store the submitted input snapshot, result, artifact ID, warnings, and generated report together in session state. Preserve it across download interactions and unrelated reruns.
- Clearly indicate when current edits differ from the assessed snapshot; never generate a report from new inputs and an old score. Add reassess and clear-session actions.
- Use an anonymous report ID by default. Make loading, service failure, invalid input, unsupported domain, missing explanation, and successful result deliberate UI states.

Done when: the report and result remain tied to the assessed snapshot; downloads do not lose the result; edits cannot silently alter a previous report; clearing a session removes all assessment data.

### HW-08 — Polish visual design and result communication

Evidence: the app already has a branded hero, cards, colors, and grouped fields. The hero calls the demo “validated”; the score caption calls calibrated output “raw”; arbitrary bands 0.35/0.65 coexist with a binary threshold of 0.5. Opening and closing HTML card tags are issued through separate `st.html` calls, which should not be assumed to wrap intervening native widgets.

Implement:

- Keep the HeartWise identity; improve spacing, hierarchy, typography, and mobile stacking using native Streamlit containers/theme configuration before adopting fragile CSS selectors.
- Put purpose and limitations near the first action. Say “internally evaluated research demo” rather than implying clinical validation.
- Make score, positive-class meaning, decision threshold, demo band, and uncertainty distinct. A 0.50 decision threshold and 0.35/0.65 bands can coexist if explained; preferably use one versioned policy shared everywhere.
- Replace contradictory “raw” wording when the served score is calibrated. Avoid implied future risk or a guaranteed healthy result.
- Add readable contribution bars/waterfall only under the explanation contract in HW-03. Provide a text alternative and distinguish direction without color alone.
- Check keyboard flow, focus, labels, contrast, zoom, mobile widths, and dark/light behavior. Self-host or use system fonts if offline operation matters.
- Organize the dashboard around questions: how evaluated, what errors occurred, what limitations apply. Move long methodological text into expanders.

Done when: browser review at desktop and narrow mobile widths shows no clipping; the entire form works by keyboard; all score/band wording agrees with policy; no unqualified validation claim remains.

### HW-09 — Make PDF reports robust and professional

Evidence: `src/report.py:69` interpolates patient ID directly into ReportLab Paragraph markup. Table cells use plain strings without deliberate wrapping. The PDF uses default Helvetica alongside arrow symbols, gives server-local time without a timezone, and lacks units in its generic input table.

Implement:

- Escape text before building Paragraphs, bound report IDs, and sanitize download filenames independently. Test markup-like and Unicode identifiers.
- Use wrapped cells, repeated table headers, sensible page breaks, a suitable embedded font, and page numbers. Verify arrow/other symbols or replace them with readable text.
- Include input values and units, positive-class semantics, calibrated/base score status, threshold/band policy, domain warnings, explanation limitations, artifact/run ID, and a timezone-aware timestamp.
- Generate from the stored assessment snapshot; do not retain PDFs server-side by default. Match report terminology to the interface.

Done when: normal, long-ID, special-character, missing-explanation, and multipage reports render without clipping or glyph failures; extracted values match the assessment; downloaded filenames are safe.

### HW-10 — Reproduce the development environment

Evidence: runtime dependencies are pinned, but the supplied local interpreter differs materially and cannot run the saved artifact. No Python version declaration or installation guide exists. Modules repeatedly mutate `sys.path`; dataset defaults depend on working directory.

Implement:

- Declare the supported Python version (currently Docker uses 3.12), project metadata, and a clean installation workflow. Verify every existing dependency pin resolves together before changing it.
- Separate runtime, training, UI, and development extras where useful; lock a tested environment. Do not change the global interpreter or silently downgrade the artifacts.
- Turn shared code into an importable package with consistent imports and explicit configuration. Resolve data/artifact paths independently of shell working directory.
- Record training environment versions and validate compatibility at startup. Fail with actionable setup instructions instead of allowing known-incompatible artifacts to serve.
- Keep explicit CLI entry points for training, data audit, evaluation, and artifact validation. Stop globally suppressing all warnings; retain important incompatibility and metric warnings.

Done when: a fresh supported environment can install and run a canary prediction; imports work from another working directory; incompatible versions produce a clear error; install/training/run steps are documented.

### HW-11 — Test contracts and failure modes

Implement an incremental test suite with small fixtures and disposable outputs. Focus on behavior, not tests that duplicate code:

- Source-verified mappings, target profiles, special codes, dataset checks, and split isolation.
- API/local input-validation parity and malformed/nonfinite payloads.
- Probability class ordering, threshold boundary behavior, domain status, and explanation reconstruction/grouping.
- Consistency of saved predictions, confusion counts, metrics, and manifests.
- API readiness, startup failure, response contracts, and remote/local mode behavior.
- Assessment snapshot persistence and download reruns, using Streamlit's testing facilities where suitable.
- PDF escaping, content fidelity, and visual rendering checks.
- A fast training smoke test on a fixture and a container service smoke test; put costly research evaluations in a separately triggered job.

Add CI for supported Python, dependency installation, formatting/linting, tests, and artifact validation. Add a Git workflow when the project is placed in a repository; do not assume one already exists.

Done when: the original semantic/report/fallback bugs would fail a regression test, routine checks run quickly, and CI reports actionable failures without retraining for every trivial change.

### HW-12 — Separate model training from deployment

Evidence: Docker's `builder` installs build tools and all dependencies, then trains; both runtime targets inherit that full layer. A cache invalidation can retrain and overwrite an artifact during deployment. Compose depends on startup order, not API readiness.

Implement:

- Provide a separate training command/job that produces a validated immutable bundle. Deployment images consume a specified bundle; make optional train-on-build a clearly named demo path if retained.
- Build slim service images from clean runtime stages; omit compilers/training-only tools where possible. Run as an unprivileged user and set writable locations explicitly.
- Make Compose wait for API health/readiness, use explicit configuration, and document local/public deployment differences. Expose only needed ports in a public setup.
- Pin and record the tested runtime/base image strategy; add build/start/inference smoke checks. Add resource limits after measuring needs.
- Document reverse-proxy/TLS setup for public hosting, rollback to a known artifact, health checks, and startup troubleshooting.

Done when: ordinary application deployment does not retrain; both services identify the same bundle; readiness gating works; builds and canary requests succeed in a clean environment.

### HW-13 — Protect data and keep claims proportionate

Evidence: `api/main.py:40` allows all CORS origins; line 76 includes raw exception text in responses. No explicit privacy/retention policy exists. Streamlit talks to FastAPI server-to-server, so browser CORS may not be needed for the current interface.

Implement:

- Configure only required CORS origins/methods or disable it when unnecessary. CORS is not authentication.
- Keep a public synthetic-demo mode simple. Add request-size limits, bounded identifiers, and deployment-appropriate throttling for an exposed inference endpoint. Protect any future admin/model-upload operations separately.
- Never load user-supplied joblib/pickle files. Validate trusted artifact hashes and storage permissions; model loading is a trust boundary.
- State what is sent to the API, what exists only in session memory, and how clearing works. Avoid logging raw clinical input, report IDs, or PDFs. Do not claim legal compliance from these controls alone.
- Clarify “disease status in this dataset” versus future cardiovascular risk. Keep a concise educational-use statement visible without overwhelming the form.
- Do not generate personalized treatment recommendations from SHAP directions or scenario changes.

Done when: errors contain no paths/stack details; logs omit patient fields; origins and limits are configurable; the privacy statement matches actual behavior; all public claims have evidence.

### HW-14 — Upgrade the evaluation dashboard and model card

Implement:

- Add specificity, confusion counts, precision-recall curve/AP, threshold curves, Brier score, and reliability plots with bin sample counts. Distinguish cross-validation dispersion from a confidence interval.
- Add bootstrap or appropriate interval estimates with stated method, seed, sample count, and small-sample limitations. Avoid excessive decimal precision.
- Show candidate comparison versus the final model, selection rules, calibration policy, data profile, training support, and known failures in a dedicated model card.
- Explain five missed positives out of 28 actual positives alongside recall; do not present percentages without denominators.
- Use interactive charts backed by saved evaluation tables when useful, while retaining exportable figures. Handle absent artifacts deliberately.

Done when: every chart and metric is reproducible from the bundle, clearly labeled, and understandable without reading the source code.

### HW-15 — Add a transparent research playground

Implement only after HW-01–05:

- A threshold explorer showing sensitivity/specificity and FP/FN tradeoffs from saved validation predictions. Clearly label it experimental; changing a slider must not silently change the default deployed policy.
- A side-by-side scenario comparison showing exactly which fields changed and whether either record lies outside training support.
- A one-variable-at-a-time response curve for supported inputs, preserving actual categories and units.
- Explain that changing a model input is not evidence that a real intervention causes that score change. Never turn the playground into treatment advice.

Done when: scenarios are validated identically to ordinary input; comparisons retain model identity; calibration and threshold policy are visible; no counterfactual causal claims appear.

### HW-16 — Assess generalization and subgroup performance

Implement:

- Evaluate sex and broad age groups with sample counts, sensitivity/specificity/calibration where estimable, and uncertainty. Suppress unstable metrics or label them clearly when denominators are tiny.
- Treat subgroup results as descriptive limitations, not proof of fairness. The dataset's demographic coverage is limited.
- Consider additional UCI hospital datasets only after auditing feature availability, encodings, missingness, cohort differences, licenses, and outcome compatibility. Keep external evaluation distinct from development data; retraining on it removes its status as an untouched external test.
- Avoid adding BMI, smoking, family history, or lifestyle sliders to this model unless a compatible dataset supports them and a new model is trained and evaluated.

Done when: external/subgroup reports document their populations and limitations, and no unsupported feature is presented as influencing the existing model.

### HW-17 — Measure serving performance and manage models

Implement:

- Measure cold startup, warm prediction, explanation latency, memory, and concurrency using synthetic inputs; report p50/p95 with the environment and workload.
- Keep loading process-scoped; remove unnecessary duplicate local warmup in API mode. If explanations dominate latency, consider a separate optional explanation request with explicit status.
- Add structured telemetry for request IDs, outcome/error class, artifact ID, and timing without patient data. Avoid caching patient predictions globally.
- Use versioned artifact directories, canary validation, atomic activation, and rollback. Expose current artifact identity in readiness/model information.
- If monitoring input distributions later, use aggregate privacy-aware metrics. Distribution drift does not measure accuracy without trustworthy delayed labels.

Done when: startup/prediction targets are chosen from measured baselines; failures are observable; a bad bundle cannot activate; rollback is exercised.

### HW-18 — Make the project easy to evaluate and present

Implement:

- Write a README with purpose, verified screenshots, supported Python, local/API/Docker setup, sample API request/response, architecture diagram, training/evaluation commands, artifact identity, and limitations.
- Add `docs/MODEL_CARD.md`, `docs/DATA_CARD.md`, deployment/troubleshooting instructions, contribution guidance, and a changelog. Choose a code license explicitly; keep dataset attribution separate.
- Build a short case study: problem, audience, data audit, leakage correction, model tradeoffs, faithful explanation design, error analysis, and lessons learned. Show before/after bugs with evidence rather than cherry-picked accuracy gains.
- Provide synthetic demo scenarios, a polished sample report, and a two-minute walkthrough. Screenshots and figures must come from the corrected current build.
- Include an honest resume description, such as: “Built an explainable research classifier with FastAPI, Streamlit, reproducible evaluation, model provenance, and patient-level explanation checks.” Adapt it to what is actually implemented.

Done when: another developer can reproduce the demo from the README and a reviewer can understand both strengths and limits within a few minutes.

### HW-19 — Optional batch and research-session features

Useful if the goal is a research workflow: CSV batch validation, row-level error feedback, bounded batch inference, downloadable results, and anonymous session-local comparisons. Define limits and avoid real identifiers by default.

Persistent history, accounts, and saved reports need explicit retention, deletion, access-control, and storage design. They are not necessary for a strong portfolio demo. Prefer session-only history first.

Done when: batch and single-record results agree; one malformed row cannot corrupt other results; memory/request bounds are enforced; no persistent patient database appears without a documented requirement.

### HW-20 — Optional larger technology changes

Stay with Streamlit until browser review shows a concrete limitation. A React/Next.js frontend can help with extensive workflows or precise interaction design, but it increases implementation and maintenance work. Likewise, MLflow/DVC may help repeated experiments; Kubernetes, microservices, a chatbot, deep learning, and mobile packaging are not prerequisites for quality here.

Choose an expansion only when it solves a measured problem. A wider algorithm collection, more animation, or a higher historical accuracy alone is not a success criterion.

## 5. Suggested delivery sequence

1. **Foundation:** HW-10 environment setup; HW-01 source semantics and provenance; regression tests for both. Minimal truthful wording corrections can accompany this phase.
2. **Scientific correctness:** HW-02 evaluation; HW-04 bundle consistency; HW-03 explanation contract. Establish trusted baseline outputs.
3. **Reliable core:** HW-05 validation; HW-06 serving; HW-07 state; HW-13 privacy/error handling. Add tests with each change.
4. **Polished experience:** HW-08 visual/browser QA; HW-09 PDF QA; HW-14 model card/dashboard.
5. **Delivery and presentation:** HW-12 deployment; HW-17 performance; HW-18 README and demonstration; complete HW-11 CI.
6. **Differentiation:** HW-15 playground and HW-16 external/subgroup evaluation. Add HW-19/20 only for a stated need.

Best small scope: verified semantics, leakage-free calibration selection, faithful explanations, consistent reports, reproducible setup, and an excellent README. Best larger scope: all of P0/P1 plus the threshold playground and carefully qualified external evaluation.

## 6. Ready-to-use Codex prompts

Use one phase per session or manageable change. The detailed acceptance criteria above are part of each prompt; ask Codex to read the file, not guess from the summary. Prompts authorize implementation within the project, but public publishing and adding persistent sensitive-data storage are separate decisions.

### Master prompt: implement the core roadmap in phases

```text
Read HEARTWISE_IMPROVEMENT_PLAN.md in this project and inspect the current code before editing. Implement the P0 and P1 roadmap in dependency order, beginning with a reproducible supported environment and HW-01. Treat the review as evidence to recheck, not an excuse to overwrite newer work. Preserve the existing FastAPI/Streamlit product unless a concrete requirement needs a change.

Work in manageable phases, add meaningful regression tests with each fix, and update docs/IMPLEMENTATION_STATUS.md with task ID, changes, validation results, and remaining limitations. Preserve the original data and trained artifacts as historical inputs; write new training outputs to versioned directories and activate only a complete validated bundle. Keep test data out of all model/calibration/threshold selection and do not optimize to match the old metrics. Explain the served model honestly and retain all grouped SHAP contributions.

Complete authorized local implementation and checks. Do not publish externally or introduce accounts, persistent patient storage, or unrelated features. If a required source, runtime, or dependency cannot be accessed, record the precise limitation and continue independent work. At each phase completion report what changed, tests run, and which acceptance criteria remain unresolved. Do not mark untested behavior complete.
```

### Phase 1 prompt: environment, labels, and source evidence

```text
Read HEARTWISE_IMPROVEMENT_PLAN.md. Implement HW-10 and HW-01 with the relevant HW-11 regression tests. Verify supported Python and dependency compatibility in an isolated project environment; do not modify the global interpreter. Correct cp/restecg/slope labels to the verified local encodings, keep the source-verified target inversion for this exact dataset profile, and correct the thallium-test terminology. Add reproducible source reconciliation, a data dictionary/manifest, and DATA_CARD documentation. Preserve raw data and old models. Replace correlation-based provenance claims with the exact evidence and make row counts dynamic. Validate every affected UI/API/report label. Record completed criteria and remaining checks in docs/IMPLEMENTATION_STATUS.md.
```

### Phase 2 prompt: credible evaluation and consistent artifacts

```text
Read HEARTWISE_IMPROVEMENT_PLAN.md and the current implementation status. Implement HW-02 and HW-04 with leakage/consistency regression tests. Move learned constraints, calibration selection, tuning, and threshold selection inside training-only folds; explicitly configure calibration behavior. Keep the final test set outside selection. Retain logistic regression as a serious baseline and generate truthful selection rationale. Write final predictions once, then derive final plots, metrics, error analysis, metadata, and dashboard manifests from them. Preserve candidate/base outputs with explicit labels. Produce a versioned validated bundle without overwriting historical artifacts. Document internal retrospective evaluation and small-sample uncertainty; do not target a promised accuracy.
```

### Phase 3 prompt: faithful explanations

```text
Read HEARTWISE_IMPROVEMENT_PLAN.md. Implement HW-03 against the evaluation/artifact design now present. Establish an explicit contract for whether explanations target the final calibrated model or a separately labeled base model. Aggregate all encoded contributions for each raw feature, retain a complete attribution vector and residual for truncated views, expose scale/base value/target model ID, and test reconstruction on the explained model's scale. Ensure actual selected categories are displayed and explanation failures are visible. Update API, interface, report, and model-card terminology together. Benchmark the chosen explanation method on synthetic inputs and record fidelity and latency results.
```

### Phase 4 prompt: reliable serving and assessment workflow

```text
Read HEARTWISE_IMPROVEMENT_PLAN.md. Implement HW-05, HW-06, HW-07, and HW-13 with behavior tests. Share strict validation between local and API paths, add training-domain warnings, explicit local/API/demo-fallback modes, real readiness checks, safe errors, and artifact-aware metadata. API-only mode must work without local models; never hide API validation errors behind fallback. Keep each assessment as one session snapshot including its inputs, result, artifact identity, warnings, and export. Preserve it on reruns/downloads, show stale edits, and add clear-session behavior. Keep privacy controls proportionate to a synthetic educational demo and do not add persistent patient storage.
```

### Phase 5 prompt: design, dashboard, and PDF polish

```text
Read HEARTWISE_IMPROVEMENT_PLAN.md. Implement HW-08, HW-09, and HW-14 while retaining HeartWise's visual identity. Build guided inputs with units/help, deliberate loading/error/unsupported states, accessible result and explanation views, and a dashboard with denominators, uncertainty, calibration evidence, and model/data cards. Generate escaped, wrapped, timezone-aware reports from the assessed snapshot with matching terminology and artifact IDs. Verify the actual interface at desktop/mobile widths and with keyboard/zoom. Render representative PDFs and inspect every page. Do not mark visual checks passed if the required tools are unavailable; document the limitation. Update implementation status and documentation.
```

### Phase 6 prompt: tested deployment and portfolio delivery

```text
Read HEARTWISE_IMPROVEMENT_PLAN.md. Implement HW-12, HW-17, HW-18, and finish HW-11. Separate training from ordinary deployment, consume versioned validated artifacts, slim runtime images, run unprivileged, and gate Compose startup on readiness. Add canary checks, artifact rollback, privacy-safe telemetry, and measured startup/latency baselines. Create CI and a complete README with reproducible setup, verified screenshots, sample requests, architecture, model/data limitations, and a short case study. Use synthetic demo inputs. Run the appropriate clean-environment/container checks and state exactly which passed. Do not publish the project externally.
```

### Optional differentiation prompt

```text
Read HEARTWISE_IMPROVEMENT_PLAN.md and confirm the core acceptance criteria are satisfied. Implement HW-15 and HW-16 as clearly labeled research features: validation-data threshold exploration, controlled scenario comparison, and qualified subgroup/external evaluation. Keep the deployed default policy unchanged unless deliberately versioned. Audit external cohorts before use, report small denominators and uncertainty, and distinguish association from intervention effects. Do not add unsupported lifestyle inputs or make clinical-validity/fairness claims. Record reproducible methods, tests, and limits.
```

## 7. Completion checklist

- [ ] Labels, target interpretation, missing codes, and dataset profiles are source-verified.
- [ ] No final-test labels influence model, constraint, calibration, or threshold selection.
- [ ] Metrics, plots, PDF, dashboard, and predictions identify one consistent artifact/policy.
- [ ] Explanations preserve grouped contributions and name their target/scale accurately.
- [ ] Validation is consistent; unsupported-domain records are handled deliberately.
- [ ] API-only mode, service failures, readiness, reruns, stale edits, and downloads are tested.
- [ ] A clean supported environment runs inference without artifact-version warnings/errors.
- [ ] Interface and representative PDFs have actual visual verification.
- [ ] Deployment does not silently retrain and rollback is demonstrated.
- [ ] README, data/model cards, privacy notes, and portfolio assets match implemented behavior.
- [ ] No accuracy, clinical-validation, fairness, or compliance claim exceeds the evidence.

## 8. Primary references for implementation

- [UCI Heart Disease dataset, original variable definitions, license, and DOI](https://archive.ics.uci.edu/dataset/45/heart+disease).
- [Original processed Cleveland data used for the row comparison](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data).
- [scikit-learn 1.8 calibration estimator documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).

Local code paths and line references describe the reviewed snapshot and may move during implementation. Recheck the current source before changing it.
