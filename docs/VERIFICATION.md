# Verification record

Checks performed locally on Windows 11, isolated CPython 3.12.13, with Docker Desktop Linux containers. No public deployment or hosted CI run was performed.

## Passed

- All original requirements resolved and installed together. Full tested dependency lock and API/UI subsets recorded; global Python unchanged. Imports, dataset loading and inference succeeded from a different working directory through the editable installation.
- Full-field UCI audit: 303 unambiguous local matches / 302 distinct source rows, zero unmatched/ambiguous rows, seven removed local rows / six distinct missing-data source records. Exact local dataset hash enforced.
- Training-only bounded selection and three nested development folds completed. Final test metrics came from one saved prediction table; model replay, class ordering, split identity and checksums validated before activation. A repeat unchanged training command reused the bundle without fitting.
- Explanation reconstruction, full 13-field grouping, actual category labels, top-five residual, threshold boundary, unsupported-age abstention and visible explanation failure checked.
- 19-case focused regression suite passed after correcting two one-class test fixtures and a real review-confirmation normalization bug. Two additional focused tests passed for two-bundle activation/rollback and request-ID preservation on safe 500 responses. A further regression verified final-test isolation of the supplemental imputation study. Total: 22 passing cases across these runs. Ruff passed. No repeated full-suite benchmark was used.
- Shared validation rejects fractional categories, booleans, numeric strings, null/nonfinite and unexpected fields in local and API paths. Readiness failure, safe errors, absent CORS, body limits, remote no-fallback and artifact identity checked.
- Streamlit AppTest verified assessment persistence, cross-workspace navigation, stale edits, review reset and clear-session behavior.
- Actual browser: 1280×900 desktop and 390×844 mobile views, keyboard Space/Enter review/submission and Tab navigation, result/explanation, evidence dashboard and a one-variable scenario comparison. Mobile page width equaled viewport width; review and explanation tables wrap. Caption opacity was corrected for contrast; dark heading changed to #c0293b (contrast 3.26:1 against #0e1117 for large text, 5.80:1 on white). Dark theme inspected via a temporary server flag. Screenshots are in `docs/screenshots`.
- Four synthetic PDF variants (normal, long markup/Unicode ID, unavailable explanation, unsupported age) rendered as eight pages with PyMuPDF. Every page was inspected in a contact sheet: no overlapping/clipped text or missing-glyph boxes. Text extraction verified escaped identifier, artifact, labels and page numbers. Unsupported font glyphs use explicit Unicode code points rather than silent squares.
- Clean Linux API/UI images built from the pinned base and lock subsets. Non-root canary ran during API build; Compose started API before UI after readiness. Both containers became healthy. The UI image contains no artifact directory, and actual server-to-server prediction, evidence, PNG, PDF, 422 validation, 413 body limit and 8 explained requests across 4 workers passed with one artifact ID.
- Measured local baseline: cold process to prediction 2.44 s; warm prediction p50/p95 3.36/3.73 ms; explained p50/p95 7.89/8.63 ms; 4-worker explained p50/p95 31.99/32.94 ms; process working set about 160 MB. Tiny synthetic workload; not a production SLA. Container smoke concurrency completed in roughly 0.11 s; initial measured container memory was about 122 MiB API and 51 MiB UI before browser workloads.
- Two disposable validated test bundles exercised activate A → B → A, then rejected a corrupted B while preserving A. Production model files and the active production pointer were not altered by this test.

## Limits and blockers

- External hospitals were audited, not used for fitting: Hungarian/Switzerland/VA have only 1/0/1 compatible complete records. No defensible external performance estimate is possible under this complete-input policy. Missing specialist values were not fabricated.
- The browser automation's Ctrl+plus action did not change viewport/DPR; a true browser 200% zoom check remains unverified. Narrow-width reflow was checked separately. This is not a full screen-reader/accessibility certification; keyboard checks covered the critical assessment path.
- CI configuration exists but hosted execution requires an owner-created repository/runner. No repository was initialized or published just to obtain a CI badge.
- One corrected production model version exists. Rollback was exercised with disposable valid bundles, not between two independent production model releases.
- Windows sandbox blocked asyncio internal loopback socket pairs, so API/AppTest and preview checks used approved unrestricted local execution. Container rebuild encountered transient DNS/TLS registry failures; the subsequent retry succeeded. These are recorded execution limitations, not suppressed model warnings.
- Starlette emits a TestClient `httpx` deprecation warning; the locked combination passed. No blanket warnings filter was introduced.
- No individual-patient probability interval, clinical validity, fairness proof or improved-accuracy claim is established.

Supplemental missingness check: a fixed logistic/sigmoid procedure compared complete-case training with train-fold-only median/mode imputation on development data, saving a separate versioned diagnostic study. Final-test IDs were excluded before constructing study X/y, and the isolation regression passed. No retraining or activation of the served bundle resulted.

Final API-only container browser check: synthetic assessment rendered with artifact hw-8f3c351bc5ead288, all 13 explanation fields, reconstruction error 2.1e-17, and the assessed PDF download control. Screenshot: docs/screenshots/container.jpg.
