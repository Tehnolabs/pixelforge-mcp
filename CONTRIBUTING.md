# Contributing to PixelForge MCP

Thank you for your interest in contributing to PixelForge MCP. This guide covers everything you need to get started.

## Development Setup

```bash
git clone https://github.com/tehnolabs/pixelforge-mcp.git
cd pixelforge-mcp
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

Requires Python 3.10 or higher.

## Code Standards

This project uses the following tools to maintain code quality:

| Tool | Purpose | Config |
|------|---------|--------|
| [black](https://black.readthedocs.io/) | Code formatting | Line length 88 |
| [ruff](https://docs.astral.sh/ruff/) | Linting | Line length 88, select E/F/I/N/W |
| [mypy](https://mypy.readthedocs.io/) | Type checking | Strict mode |

Run all checks before submitting a PR:

```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Running Tests

```bash
# Unit tests (no credentials needed)
pytest tests/unit/ -v

# A specific test file
pytest tests/unit/test_config.py -v

# Integration tests (live API calls; skipped without credentials)
pytest tests/integration -q -m integration
```

### Integration tests

`tests/integration/` calls the live Google APIs: one request per tool with the cheapest options, about six requests per run. Without credentials every integration test is skipped. Provide one of:

```bash
# Gemini API key
export GOOGLE_API_KEY="your-api-key-here"

# Vertex AI: a service-account key, or `gcloud auth application-default login`
export GOOGLE_CLOUD_PROJECT="my-project"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
export GOOGLE_CLOUD_LOCATION="global"
```

When both are set the API key wins unless `GOOGLE_GENAI_USE_VERTEXAI=true`.

## Continuous Integration

`.github/workflows/ci.yml` runs two jobs:

| Job | Runs on | What it does |
|-----|---------|--------------|
| `test` | every push and pull request | ruff, black, mypy, unit tests on Python 3.10, 3.12, 3.14 |
| `integration` | push to `main`, `workflow_dispatch`, pull requests from branches in this repository | integration tests on Python 3.12 |

The integration job uses one of these optional repository secrets. With neither set it prints a notice and passes.

| Secret | Value |
|--------|-------|
| `GCP_SERVICE_ACCOUNT_KEY` | JSON key of a service account with the **Vertex AI User** role, stored as a single-line (minified) JSON string so GitHub's per-line secret masking covers it. Selects the Vertex AI backend with `GOOGLE_CLOUD_LOCATION=global`. |
| `GCP_PROJECT_NUMBER` | Optional. The project number, masked in the log. The job also resolves it with `gcloud projects describe` when the service account may read the project. |
| `GOOGLE_API_KEY` | Gemini API key. Used only when `GCP_SERVICE_ACCOUNT_KEY` is not set. |

Pull requests from forks cannot read repository secrets, so they run the unit checks only. In Vertex mode the job masks the project ID, the project number, and the service account email in the log, and pytest runs with `--show-capture=no` so captured logs are not echoed on failure.

## Project Structure

```
pixelforge-mcp/
├── src/pixelforge_mcp/
│   ├── config.py         # Pydantic config, VertexConfig
│   ├── server.py         # FastMCP server with 19 async tools
│   ├── templates/        # Prompt template YAML files
│   └── utils/
│       ├── api_client.py # Direct google-genai SDK client
│       ├── history.py    # Generation history tracking
│       ├── templates.py  # Template library
│       ├── transforms.py # Image transforms (crop, resize, etc.)
│       └── validation.py # Input validation with Pydantic
├── tests/
│   └── unit/             # 237 unit tests
├── config/               # Configuration examples
└── docs/                 # Documentation
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Make your changes following the code standards above
3. Add or update tests for any new functionality
4. Ensure all checks pass:
   - `black --check src/ tests/` (formatting)
   - `ruff check src/ tests/` (linting)
   - `mypy src/` (type checking)
   - `pytest tests/unit/ -v` (tests)
5. Submit a pull request with a clear description of the changes

## Reporting Issues

Found a bug or have a feature request? Please open an issue on [GitHub Issues](https://github.com/tehnolabs/pixelforge-mcp/issues).

When reporting bugs, include:
- Python version (`python --version`)
- PixelForge MCP version (`pipx list | grep pixelforge`)
- Steps to reproduce the issue
- Expected vs actual behavior

## License

PixelForge MCP is licensed under [AGPL-3.0](LICENSE). By contributing, you agree that your contributions will be licensed under the same license.
