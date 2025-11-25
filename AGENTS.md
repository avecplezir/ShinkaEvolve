# Repository Guidelines

## Project Structure & Module Organization
- `shinka/`: Core package (evolution runner, launchers, database, edit utilities, WebUI assets) plus CLI entrypoints `shinka_launch` and `shinka_visualize`.
- `configs/`: Hydra configs for tasks, evolution, database, cluster, and variants; add overrides here instead of hard-coding.
- `examples/`: Reference problems with `initial.py` and `evaluate.py` templates.
- `tests/`: Pytest suite for patch/edit helpers; add new `test_*.py` files next to fixtures.
- `docs/`: How-to guides and configuration reference; keep user-facing instructions in sync with CLI flags.
- `results/`: Auto-generated experiment outputs; avoid committing large artifacts.

## Build, Test, and Development Commands
- Env + install: `uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[dev]"`.
- Run example: `shinka_launch variant=circle_packing_example` (writes to `results/`).
- Visualize: `shinka_visualize --port 8888 --open` from project root.
- Tests: `pytest -q` or target a case with `pytest tests/test_edit_base.py::test_edit`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, type hints on public functions; keep helpers small and side-effect free.
- Preserve EVOLVE-BLOCK markers; only mutate code inside marked regions when patching examples.
- Lint/format before pushing: `black shinka tests`, `isort shinka tests`, `flake8 shinka tests` (defaults from dev deps).
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`, Hydra keys `lower_snake`.

## Testing Guidelines
- Use `pytest` with descriptive test names; place regression cases near the feature under test.
- Cover both expected flow and boundary conditions (indentation handling, EVOLVE-BLOCK validation in `shinka.edit`).
- No strict coverage gate, but add a test for every bug fix or new config surface; keep tests deterministic.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subjects; optional scope prefix is common (e.g., `llm: Add GPT-5.1 models`).
- PRs: include intent summary, linked issue/experiment config, commands/tests run, and screenshots/logs for WebUI changes.
- Keep changes scoped; update relevant docs (`docs/`, `README.md`) when behavior or CLI flags change.

## Security & Configuration Tips
- Store API keys in `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.); never commit secrets or cloud credentials.
- Prefer new Hydra overrides in `configs/variant/` instead of editing shared defaults; document required environment variables.
- Review generated artifacts before committing to avoid leaking user data or oversized result files.
