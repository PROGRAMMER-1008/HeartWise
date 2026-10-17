# Contributing

Use Python 3.12 and an isolated editable installation. Keep raw data and historical artifacts unchanged. Run source audit, focused pytest cases and Ruff before proposing a change. Routine CI validates the included bundle without retraining; container checks are manually triggered.

For model changes, predeclare development objectives/configuration, keep final-test labels out of selection, create a new versioned bundle, validate it, and record evidence in the status/model cards. Never tune against a previously reported test result or imply that a reshuffle creates external evidence. Changes to mappings require a new audited profile; encoded meanings cannot silently migrate.

For UI/report changes, use synthetic records, check snapshots across reruns/navigation, inspect narrow and desktop views, and render every representative PDF page. Do not commit patient inputs, identifiers, downloads or logs. Keep schema, API, UI and report terminology aligned.

Suggested Git workflow once the owner initializes a repository: feature branch, focused commits, review with evidence, CI, then deliberate activation of a validated artifact. This supplied folder was not a Git repository; no history was invented and nothing was published.
