# Dreamdeck: Your AI-Powered Storytelling Companion

Welcome to Dreamdeck! This application is your gateway to creating and managing immersive, collaborative stories with the help of AI. Whether you're a game master (GM) or a player, Dreamdeck will help you craft dynamic and engaging narratives, providing a Holodeck-like experience.

## Slash Commands

| Command        | Description                                                      | Example Usage                                  |
|----------------|------------------------------------------------------------------|------------------------------------------------|
| `/roll`        | Roll dice via natural language or dice notation                  | `/roll 2d6`, `/roll check perception`          |
| `/search`      | Perform a web search                                             | `/search history of dragons`                   |
| `/todo`        | Add a TODO item                                                  | `/todo Remember to buy milk`                   |
| `/write`       | Directly prompt the writer agent                                 | `/write The wizard casts a spell`              |
| `/storyboard`  | Generate storyboard images for the last Game Master message      | `/storyboard`                                 |

- **Slash commands bypass the decision agent** and directly invoke the relevant tool or agent.
- `/storyboard` generates images for the **last Game Master message**.
- `/todo` saves reminders and notes in local markdown files under `./helper/YYYY-MM-DD/todo.md`.

## New Features

- **Smart Categorization of User Input:**
  - The app automatically classifies normal chat input into:
    - **Roll:** If the user wants to roll dice, the app will handle the roll before continuing the story.
    - **Search:** If the user wants to look up information, the app will perform a web search using SerpAPI.
    - **Continue Story:** If the user is continuing the story, the app will generate the next part of the narrative.
  - **Slash commands bypass this classification** and directly invoke the relevant tool.

## Overview

Dreamdeck is designed to:
- **Generate Immersive Narratives:** The AI Game Master (GM) creates detailed, character-driven stories that adapt to your choices and preferences.
- **Visualize Scenes:** Generate vivid storyboards to bring your narrative to life.
- **Store Everything Locally:** All your stories, characters, and elements are stored locally, ensuring your data is safe and accessible.
- **Support Multiple Languages:** Enjoy the app in multiple languages, including English, Bengali, Gujarati, Hebrew, Hindi, Kannada, Malayalam, Marathi, Dutch, Tamil, Telugu, and Chinese (Simplified).

## Getting Started

### Start a New Chat

1. **Open the App:**
   - When you first open the app, you'll see a welcome message from the Game Master. Choose the type of story you want to create.
   - The Game Master will guide you through the narrative, responding to your inputs and creating a dynamic, collaborative adventure.

### Interact with the GM

- **Describe Your Actions:** Type your actions or decisions in the chat box. The GM will respond with rich, detailed descriptions and narrative developments.
- **Explore Different Tones:** The GM can adapt to your preferred tone, whether it's lighthearted, serious, or intense. Just let the GM know how you want the story to feel.
- **Handle NPCs:** The GM will introduce and manage non-player characters (NPCs) with distinct personalities and motivations. Engage with them to uncover new story elements.
- **Avoid Speaking for the Player:** The GM will never assume your actions, thoughts, or emotions. You have full control over your character's decisions.

### Generate Storyboards

- **Visualize Scenes:** After the GM generates a narrative segment, the app will create storyboards to bring the scene to life.
- **View and Save Images:** The generated images will be displayed in the chat. You can save them for future reference or share them with others.
- You can also use `/storyboard` to generate images for the **last Game Master message** at any time.

### Use the Dice Roll Tool

- **Add an Element of Chance:** Use the `/roll` command to roll dice with natural language or dice notation.
- Dice rolls are also detected automatically in normal chat input.

### Manage TODOs

- Use `/todo` to save reminders and notes.
- TODO items are saved locally in markdown files under `./helper/YYYY-MM-DD/todo.md`.

### Knowledge Management

- **Location**: The knowledge directory MUST reside at the root of your project (`./knowledge`). 
- **Container Sync**: When running in Docker, ensure the `knowledge` folder is mounted or copied into the container.
- Files are **chunked and indexed** on startup or chat resume.
- The knowledge base is used for **retrieval-augmented generation** to enrich the story.

### Troubleshooting
If the system reports the knowledge directory is missing:
1. Confirm the `knowledge` folder exists in your project root.
2. Check Docker builds include the folder (see Dockerfile edits above).
3. Verify the path in `config.yaml` is set to `"knowledge"` (without leading slash).
4. Restart services after making changes.

### Example Usage

- **Rolling Dice:**
  - **User:** "I want to roll a 20-sided die."
  - **App:** 🎲 You rolled a 15 on a 20-sided die.
  - **Or:** `/roll 1d20`

- **Web Search:**
  - **User:** "Search for the history of dragons in mythology."
  - **App:** Summarized search results.
  - **Or:** `/search history of dragons`

- **Continuing the Story:**
  - **User:** "I approach the old wizard in the tower."
  - **App:** The wizard looks up from his ancient tome, his eyes twinkling with curiosity. "Ah, a visitor! What brings you to my humble abode?"

### Local Storage

All your stories, characters, and elements are stored locally in a PostgreSQL database. This ensures that your data is safe and accessible, and you can resume your adventures at any time.

### Support and Feedback

If you have any questions, suggestions, or encounter any issues, feel free to open an issue in the repository or reach out to the community. We're here to help you create the best possible narrative experiences!

### Community

Join our community to share your stories, get help, and collaborate:
- **GitHub Issues:** Report bugs, request features, and discuss ideas.
- **Discord:** Join our Discord server for real-time support and collaboration.

---

Dreamdeck is your portal to endless storytelling. Dive into the world of immersive narratives and bring your stories to life with the power of AI. Happy adventuring!
