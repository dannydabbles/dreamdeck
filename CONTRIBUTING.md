# Contributing to Dreamdeck

Thank you for your interest in contributing to Dreamdeck! We welcome contributions of all kinds, including bug fixes, new features, documentation improvements, and more.

## How to Contribute

1. **Fork the Repository**

   Click the "Fork" button on GitHub and clone your fork locally:

   ```bash
   git clone https://github.com/your-username/dreamdeck.git
   cd dreamdeck
   ```

2. **Create a Branch**

   Create a new branch for your feature or fix:

   ```bash
   git checkout -b my-feature
   ```

3. **Set Up Your Environment**

   - Install [Poetry](https://python-poetry.org/)
   - Run `make install` to install dependencies
   - Use `make run` to start the app locally
   - Use `make test` to run tests

4. **Understand the Refactored Architecture**

   - **Agents:** All agents (writer, dice, search, TODO, knowledge, storyboard, report, persona classifier) are stateless async functions, registered in `src/agents/registry.py`.
   - **Persona System:** Persona agents are dynamically registered and routed via the supervisor, with prompt selection based on persona and config.
   - **Supervisor:** The supervisor agent in `src/supervisor.py` routes user input to the correct persona agent or tool.
   - **Chainlit Integration:** All event handlers, commands, and UI elements are registered via Chainlit decorators in `src/app.py` and `src/event_handlers.py`.
   - **Prompts:** All prompts are Jinja2 templates in `src/prompts/` and mapped in `config.yaml`.
   - **Testing:** Tests are organized into `tests/smoke/` (fast, isolated) and `tests/integration/` (multi-component, LLM-involving). Use mocks and fixtures for LLMs, Chainlit context, and vector stores.

5. **Make Your Changes**

   - Follow existing code style and conventions.
   - Add or update tests as needed (see `tests/README.md`).
   - Update documentation if applicable.
   - If adding a new agent or persona, register it in `src/agents/registry.py` and add a prompt template if needed.

6. **Run Tests**

   Ensure all tests pass before submitting:

   ```bash
   make test
   ```

7. **Commit and Push**

   ```bash
   git add .
   git commit -m "Describe your changes"
   git push origin my-feature
   ```

8. **Open a Pull Request**

   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Describe your changes clearly
   - Link related issues if any

## Code Style

- Use [Black](https://black.readthedocs.io/) for formatting (`make format`)
- Follow PEP8 guidelines
- Write clear, concise docstrings
- Keep functions small and focused
- Register new agents/tools in `src/agents/registry.py`
- Add or update prompt templates in `src/prompts/` and map them in `config.yaml`

## Reporting Issues

If you find a bug or have a feature request, please open an issue with:

- A clear title and description
- Steps to reproduce (if a bug)
- Expected vs actual behavior
- Screenshots or logs if helpful

## Community

Join our Discord or GitHub Discussions to chat with other contributors!

Thank you for helping make Dreamdeck better!
