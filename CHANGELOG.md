# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Major Refactor

- **Modular, Stateless Agents:** All agents (writer, dice, search, TODO, knowledge, storyboard, report, persona classifier) are now stateless async functions, decorated for Chainlit and LangGraph compatibility.
- **Central Agent Registry:** Agents/tools are registered in `src/agents/registry.py` for dynamic routing, CLI, and extensibility.
- **Persona System:** Persona agents are dynamically registered and routed via the supervisor, with prompt selection based on persona and config.
- **Supervisor Orchestration:** The supervisor agent routes user input to the correct persona agent or tool, using classifier suggestions and user settings.
- **Chainlit Integration:** All event handlers, commands, and UI elements are registered via Chainlit decorators in `src/app.py` and `src/event_handlers.py`.
- **Improved Testability:** All agents and workflows are easily monkeypatchable for fast, isolated, and integration tests. Test mode is supported via the `DREAMDECK_TEST_MODE` environment variable.
- **Expanded Slash Commands:** Added `/report`, `/persona`, `/help`, `/reset`, `/save` commands.
- **Extensible Prompts:** Persona and tool prompts are now Jinja2 templates, mapped in config and loaded at startup.
- **Knowledge Base Loading:** Knowledge documents are loaded, chunked, and indexed on startup or chat resume, with improved error handling and metadata tagging.
- **Improved CLI:** The CLI can list agents, run any agent or workflow, and export state.

### Core Features

- AI-powered storytelling with GPT models
- Slash commands: `/roll`, `/search`, `/todo`, `/write`, `/storyboard`, `/report`, `/persona`, `/help`, `/reset`, `/save`
- Knowledge base loading from `./knowledge`
- Dice rolling with natural language
- Web search integration via SerpAPI
- Storyboard image generation via Stable Diffusion
- Local vector store with ChromaDB
- Export story as markdown
- Reset and save commands
- Multi-agent orchestration
- Configurable via `config.yaml` and environment variables
- Extensive test suite with pytest and mocks

## [Future]

- Multi-user support
- Player profiles and preferences
- Undo/redo message support
- `/load` command for saved stories
- More advanced knowledge base management
- Improved UI customization
- Plugin system for new tools and agents
