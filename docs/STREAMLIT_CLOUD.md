# Deploy to Streamlit Community Cloud

This deployment runs the UI and model together in local mode. No separate API,
Docker service, training job, or API key is required.

1. Upload this project to a GitHub repository, preserving folder structure.
   Include `app/`, `src/`, `artifacts/active.txt`, the complete
   `artifacts/hw-8f3c351bc5ead288/` folder, `.streamlit/config.toml`,
   `requirements-ui.lock`, `pyproject.toml`, and `packages.txt`.
   Do not upload virtual environments, caches, or secrets. The `.gitignore`
   file excludes those when using Git.
2. Sign in at https://share.streamlit.io and connect the GitHub account.
3. Choose Create app and select the repository and the branch you uploaded.
4. Set the main file path to `app/streamlit_app.py`.
5. Open Advanced settings and select Python **3.12**. The default Python
   version may differ; this model validates the Python version at startup.
6. Leave secrets empty and deploy. Local serving is the default.

Community Cloud discovers `app/requirements.txt` before the root dependency
files. It installs the pinned UI runtime plus XGBoost, which is required by the
artifact's dependency validation even though the selected model is logistic
regression. The editable project installation makes `src` importable.
The root `packages.txt` provides the Linux OpenMP runtime.

After deployment, check Synthetic A and Synthetic B, complete review and
assessment, download a PDF, and open Model evidence. If startup reports a
runtime mismatch, inspect the build log for failed package installation and
verify Python 3.12. Do not retrain or change version pins to bypass validation.

Use synthetic records for the public demonstration, as described in README.md.
This guide prepares deployment; a successful cloud build and smoke check are
still needed before claiming the hosted app works.
