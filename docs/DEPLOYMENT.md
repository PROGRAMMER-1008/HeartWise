# Deployment and troubleshooting

The Dockerfile consumes a specific validated immutable bundle. Training is never part of an image build. Base: Python 3.12.13 slim Bookworm, pinned digest `sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2`. Update this deliberately for security maintenance and re-run canary checks. Service dependency lock subsets come from the tested environment. CPU-only images omit the optional CUDA NCCL library; no GPU execution is supported. No compiler toolchain, SHAP/Numba or training plots are required at runtime.

```powershell
$env:HEARTWISE_ARTIFACT=(Get-Content artifacts/active.txt).Trim()
docker compose up --build --wait --wait-timeout 120
docker compose ps
docker compose exec -T streamlit python -c "from src.client import ServingClient; from src.explain import DEMO; c=ServingClient(); print(c.predict(DEMO)['artifact_id'])"
docker compose down
```

UID 10001 runs both services. Compose uses read-only roots and writable temporary mounts. API is internal only; the UI binds `127.0.0.1:8502`. UI image contains no artifact folder: its model identity comes from `/model-info`. API readiness runs a real synthetic inference. Streamlit waits for API health before startup.

For future public hosting, put a maintained reverse proxy with TLS in front of the UI, retain Streamlit XSRF protections, enforce connection/body/time limits and appropriate authentication. Do not expose an unauthenticated service to real clinical data. The built-in 120 requests/minute limit is per API worker, global rather than per-user, and is only a basic demo safeguard; production throttling belongs at the proxy. CORS is disabled unless `HEARTWISE_CORS_ORIGINS` is set and is never authentication. No public deployment was performed.

Configuration: `HEARTWISE_MODE=local|api`, `HEARTWISE_API_URL`, `HEARTWISE_TIMEOUT_SECONDS` (15), `HEARTWISE_ARTIFACT` (overrides active pointer), `HEARTWISE_MAX_BODY_BYTES` (8192), `HEARTWISE_REQUESTS_PER_MINUTE` (120), optional CORS origins separated by commas. Avoid request-body logging in reverse proxies, APM, and hosting infrastructure.

Activation/rollback:

```powershell
.venv/Scripts/python.exe -m src.artifacts validate hw-8f3c351bc5ead288
.venv/Scripts/python.exe -m src.artifacts activate hw-8f3c351bc5ead288
```

To roll back, pass a previously validated version ID to the same command and restart the API, or rebuild Compose with that ID. Every activation rechecks the bundle and inference canary before atomic pointer replacement. Never point the new runtime at legacy unbundled model files. One corrected validated model version currently exists; cross-version production rollback cannot be claimed from historical incompatible files. A disposable two-bundle lifecycle regression exercises activation and return to the earlier bundle without modifying production artifacts.

Troubleshooting:

- 503 readiness: verify Python 3.12 and exact bundle estimator-library versions, checksum errors, configured artifact ID and permissions. `python -m src.artifacts validate` gives the operator the diagnostic. Client errors intentionally omit paths.
- Missing active pointer: activate a validated bundle explicitly. Do not train during startup.
- Staging directory exists: inspect the failed directory; preserve it for diagnosis or move it to a clearly named quarantine before retrying. The service will not load it.
- API-only UI fails: verify the URL from the Streamlit server, not the browser. No local fallback hides API validation or server failures.
- Windows sandbox API/AppTest hangs: asyncio may block creating its internal loopback socket pair. Run in a normal permitted terminal. This was an execution-sandbox limitation, not an application workaround.
- `httpx` TestClient deprecation: the tested Starlette release accepts it but recommends `httpx2`; retained compatible locked versions, no blanket warning suppression.
- Changing model files in place: unsupported. Create and validate a new bundle; restart the API to avoid mixed model identities.

Measured Windows process working set was about 160 MB under a tiny synthetic workload. Container memory includes runtime/framework overhead; choose production resource limits only after measuring the actual concurrency and deployment environment.
