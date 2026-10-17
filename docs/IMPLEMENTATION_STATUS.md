# Implementation status

Scope: HW-01 through HW-18; HW-19/20 excluded. Original CSV and historical model/report artifacts preserved. Legacy replaced source snapshots are under `docs/historical`. No public publication, global Python changes or persistent patient storage.

Active validated bundle: **hw-8f3c351bc5ead288** (sigmoid-calibrated logistic regression C=0.1). Full evidence and execution limitations: [VERIFICATION.md](VERIFICATION.md).

| ID | Status | Changes and verification |
|---|---|---|
| HW-01 | Complete | Full-field source reconciliation 303/302; corrected labels; profile/hash guard; mapping manifest/data card; regression fixtures pass. |
| HW-02 | Complete | Bounded complete-procedure training-only selection, nested development CV, fixed threshold, no learned constraints; fold isolation tests; one final retrospective test table. |
| HW-03 | Complete | Final-probability raw-field Shapley estimates; 13 contributions, background/scale/target/residual and reconstruction; failure state and latency checked. |
| HW-04 | Complete | Immutable staging/validated activation; final figures/metrics from saved predictions; checksums, split identity, replay and corruption tests. |
| HW-05 | Complete | Shared strict schema and bounded normalized IDs; type/nonfinite/category parity; support warnings and age abstention in API/UI/PDF. |
| HW-06 | Complete | Explicit local/API modes; no fallback; lifespan/canary readiness; safe typed responses; real API-only container serving verified without local artifacts. |
| HW-07 | Complete | Guided measured/synthetic records, review step, durable snapshot/PDF, stale edits, clear session and cross-page retention; AppTest and browser checks. |
| HW-08 | Implemented; limited check remains | Native responsive form, truthful results, wrapping tables and contrast fixes; desktop/mobile/dark/critical keyboard path verified. True 200% browser zoom not supported by the available automation action. |
| HW-09 | Complete | Escaped wrapped embedded-font PDFs with UTC, units, policy/domain/artifact/explanation; four variants/eight pages rendered and inspected, extraction tests pass. |
| HW-10 | Complete | Isolated Python 3.12.13, original pins verified, tested locks/extras/package/CLIs; other-working-directory inference and clean Linux container canary pass. |
| HW-11 | Complete locally | 22 focused passing cases across suite/targeted runs; Ruff and CI workflow. Hosted CI not run (no remote repository/runner was created). |
| HW-12 | Complete locally | No-training image build, pinned base, cached runtime layers, non-root/read-only services, readiness-gated Compose; builds/start/real inference pass. No external deployment. |
| HW-13 | Complete | No default CORS; body/rate limits; safe errors/request IDs and privacy-safe timing logs; trusted checksum-validated joblib boundary; privacy/scope wording. |
| HW-14 | Complete | Final confusion counts, specificity/AP/Brier, ROC/PR/reliability figures/bin counts, bootstrap intervals; candidate/nested evaluation and model/data cards tied to serving artifact. |
| HW-15 | Complete | Development OOF threshold explorer; validated single-field scenario comparison/response curve with fixed deployed policy, domain warnings and artifact identity; actual browser comparison checked. |
| HW-16 | Subgroups complete; external evaluation blocked by data | Sex/age descriptive reports with tiny-denominator suppression and uncertainty. Audited three original cohorts; only 1/0/1 compatible complete rows, so no external accuracy estimate claimed. |
| HW-17 | Complete locally | Process-scoped loads, optional explanations, telemetry, measured startup/warm/memory/concurrency, synthetic canaries, atomic activation/rollback test; no global patient cache. |
| HW-18 | Complete | README, verified screenshots/sample PDF, cards, deployment/troubleshooting, MIT code license/separate data attribution, contribution/changelog/case study/walkthrough. |

Remaining limitations: external performance is not estimable from the audited complete cases; true 200% zoom and hosted CI remain unverified; rollback uses disposable release fixtures because only one corrected production version exists. Evaluation remains internal retrospective, not clinical validation. A separately versioned development-only imputation diagnostic was completed; it did not change the served complete-input policy or use the final test.

Run now: `.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py` (local mode), or set `HEARTWISE_ARTIFACT` from `artifacts/active.txt` and run `docker compose up --build --wait` (API-only UI at http://127.0.0.1:8502). See README for fresh installation and API commands.
