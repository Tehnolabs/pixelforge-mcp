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
# Run all unit tests
pytest tests/unit/ -v

# Run a specific test file
pytest tests/unit/test_config.py -v

# Run integration tests (requires GOOGLE_API_KEY)
pytest tests/integration/ -v
```

## Project Structure

```
pixelforge-mcp/
├── src/pixelforge_mcp/
│   ├── config.py         # Pydantic-based configuration
│   ├── server.py         # FastMCP server with 5 async tools
│   └── utils/
│       ├── api_client.py # API client using gemini-imagen library
│       └── validation.py # Input validation with Pydantic
├── tests/
│   └── unit/             # Unit tests
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
