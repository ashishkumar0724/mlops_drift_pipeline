# Security Policy

## Credentials
- Kaggle API keys: store in `.env` locally, GitHub Secrets in CI/CD
- Never commit `.env` or hardcode secrets
- Rotate keys quarterly via Kaggle account settings

## Dependency Security
- All packages pinned in `pyproject.toml` + `uv.lock`
- Run `uv pip compile --upgrade` monthly for security patches
- GitHub Dependabot enabled for automated alerts

## Data Handling
- Sample datasets only; no PII in repo
- Drift reports sanitized before upload
