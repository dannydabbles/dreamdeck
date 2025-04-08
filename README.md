# Dreamdeck: Your AI-Powered Storytelling Companion

Dreamdeck is an AI-powered application designed to provide a Holodeck-like experience, enabling you to create and manage immersive, collaborative stories. Whether you're a game master (GM) or a player, Dreamdeck will guide you through dynamic and engaging narratives, generate vivid storyboards, and store all your data locally for safety and accessibility.

---

## Features

- **Immersive Narratives:** The AI Game Master (GM) creates detailed, character-driven stories that adapt to your choices and preferences.
- **Vivid Storyboards:** Generate visual storyboards to bring your narrative to life.
- **Knowledge Base:** Load `.txt`, `.md`, and `.pdf` files from `./knowledge` directory, which are automatically chunked and indexed for retrieval-augmented generation.
- **Dice Rolling:** Supports natural language dice rolls, e.g., "roll 2d6 for attack" or `/roll 2d6`.
- **Web Search:** Use `/search` or natural language to perform web searches via SerpAPI.
- **Slash Commands:** Control the app with `/roll`, `/search`, `/todo`, `/write`, `/storyboard`.
- **Local Storage:** All your stories, characters, and elements are stored locally.
- **Multi-Language Support:** English, Bengali, Gujarati, Hebrew, Hindi, Kannada, Malayalam, Marathi, Dutch, Tamil, Telugu, Chinese (Simplified).
- **Flexible Tone:** The GM adapts to your preferred tone, from lighthearted to intense.
- **Rich Descriptions:** Vivid environments, cultures, and characters.
- **Adaptation and Improvisation:** "Yes, and…" or "Yes, but…" techniques to evolve the plot.
- **Clear Decision Points:** Present meaningful choices.
- **Neutral Ethical Stance:** The GM remains neutral, following your lead.

---

## Commands

| Command        | Description                                                      | Example Usage                                  |
|----------------|------------------------------------------------------------------|------------------------------------------------|
| `/roll`        | Roll dice via natural language or dice notation                  | `/roll 2d6`, `/roll check perception`          |
| `/search`      | Perform a web search                                             | `/search history of dragons`                   |
| `/todo`        | Add a TODO item                                                  | `/todo Remember to buy milk`                   |
| `/write`       | Directly prompt the writer agent                                 | `/write The wizard casts a spell`              |
| `/storyboard`  | Generate storyboard images for the last Game Master message      | `/storyboard`                                 |

---

## Getting Started

### Prerequisites

- Python 3.8+
- Docker
- Poetry (for dependency management)
- **Chainlit 2.x** (this app is built on Chainlit)

### Installation

1. **Clone the Repository:**
   ```sh
   git clone <repository-url>
   cd dreamdeck
   ```

2. **Install Dependencies:**
   ```sh
   make install
   ```

3. **Start the Application:**
   - **Locally:**
     ```sh
     make run
     ```
     Access at `http://localhost:8081`.
   - **With Docker:**
     ```sh
     make start
     ```
     Access at `http://localhost:8080`.

### Login Credentials

Default users (matching password callback):

- `admin` / `admin`
- `test` / `test`
- `guest` / `guest`

---

## Knowledge Base

- Place `.txt`, `.md`, or `.pdf` files inside the `./knowledge` directory.
- On startup or chat resume, these files are **automatically loaded**, chunked, and indexed into the vector store.
- The knowledge base is used for **retrieval-augmented generation** to enrich the story.

---

## Configuration

- The app is configured via `config.yaml`.
- You can override any config value with environment variables prefixed by `APP_`.
- Key sections:
  - `llm`: Base LLM parameters.
  - `agents`: Per-agent LLM configs.
  - `prompts`: Maps prompt keys to Jinja2 template files.
  - `features`: Enable/disable dice, web search, image generation.
  - `knowledge_directory`: Path to knowledge base folder.
- Prompts are Jinja2 templates located in `src/prompts/`.

---

## Testing

- Run all tests with:
  ```sh
  make test
  ```
- Tests use **mocks and fixtures** to simulate LLMs, Chainlit context, and vector stores.
- Coverage includes commands, agents, event handlers, and knowledge loading.

---

## Development

- **Run the App Locally:**
  ```sh
  make run
  ```
- **Run Tests:**
  ```sh
  make test
  ```
- **Lint and Format Code:**
  ```sh
  make lint
  make format
  ```

---

## Example Usage

- **Rolling Dice:**
  - User: `"I want to roll a 20-sided die."` or `/roll 1d20`
  - App: 🎲 You rolled a 15 on a 20-sided die.

- **Web Search:**
  - User: `"Search for the history of dragons in mythology."` or `/search history of dragons`
  - App: Summarized search results.

- **Continuing the Story:**
  - User: `"I approach the old wizard in the tower."`
  - App: The wizard looks up from his tome, eyes twinkling. "Ah, a visitor!"

---

## TODOs and Quick Wins

- **Gameplay:**
  - [ ] `/help` command listing all slash commands.
  - [ ] `/reset` command to clear chat history.
  - [ ] `/save` and `/load` commands for story snapshots.
  - [ ] Support multiple dice expressions in `/roll`.
  - [ ] Undo last message feature.
  - [ ] Export chat to markdown or JSON.
  - [ ] Player profiles with saved preferences.
  - [ ] Multi-user support in one story.

- **Developer Experience:**
  - [ ] Add `pyrightconfig.json` or `mypy.ini` for static typing.
  - [ ] Add `ruff` or `flake8` config.
  - [ ] GitHub Actions CI workflow.
  - [ ] Pre-commit hooks for linting/formatting.
  - [ ] Docstrings for all public functions.
  - [ ] API docs generation.
  - [ ] Sample knowledge files.
  - [ ] Sample `.env` file.
  - [ ] `CONTRIBUTING.md` guidelines.
  - [ ] `CHANGELOG.md`.
  - [ ] `LICENSE` file (MIT or Apache-2.0).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

---

## Community

Join our community:

- **GitHub Issues:** Bug reports, feature requests.
- **Discord:** Real-time support and collaboration.

---

Dreamdeck is your portal to endless storytelling. Dive in and bring your stories to life with AI!
