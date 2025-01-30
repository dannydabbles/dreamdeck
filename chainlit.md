# Dreamdeck: Your AI-Powered Storytelling Companion

Welcome to Dreamdeck! This application is your gateway to creating and managing immersive, collaborative stories with the help of AI. Whether you're a game master (GM) or a player, Dreamdeck will help you craft dynamic and engaging narratives.

## Overview

Dreamdeck is designed to:
- **Generate Immersive Narratives:** The AI Game Master (GM) creates detailed, character-driven stories that adapt to your choices and preferences.
- **Visualize Scenes:** Generate vivid storyboards and images to bring your narrative to life.
- **Store Everything Locally:** All your stories, characters, and elements are stored locally, ensuring your data is safe and accessible.
- **Support Multiple Languages:** Enjoy the app in multiple languages, including English, Bengali, Gujarati, Hebrew, Hindi, Kannada, Malayalam, Marathi, Dutch, Tamil, Telugu, and Chinese (Simplified).

## Getting Started

### Installation

1. **Clone the Repository:**
   ```sh
   git clone <repository-url>
   cd dreamdeck
   ```

2. **Install Dependencies:**
   ```sh
   poetry install
   ```

3. **Start the Application:**
   ```sh
   docker compose up -d
   chainlit run src/app.py
   ```

4. **Access the App:**
   Open your browser and navigate to `http://localhost:8080`.

### How It Works

1. **Start a New Chat:**
   - When you first open the app, you'll see a welcome message from the Game Master. You can choose the type of story you want to create.
   - The Game Master will guide you through the narrative, responding to your inputs and creating a dynamic, collaborative adventure.

2. **Interact with the GM:**
   - **Describe Your Actions:** Type your actions or decisions in the chat box. The GM will respond with rich, detailed descriptions and narrative developments.
   - **Explore Different Tones:** The GM can adapt to your preferred tone, whether it's lighthearted, serious, or intense. Just let the GM know how you want the story to feel.
   - **Handle NPCs:** The GM will introduce and manage non-player characters (NPCs) with distinct personalities and motivations. Engage with them to uncover new story elements.
   - **Avoid Speaking for the Player:** The GM will never assume your actions, thoughts, or emotions. You have full control over your character's decisions.

3. **Generate Storyboards:**
   - After the GM generates a narrative segment, the app will create storyboards to visualize the scene. These storyboards are generated using stable diffusion prompts, ensuring a vivid and immersive experience.
   - **View and Save Images:** The generated images will be displayed in the chat. You can save them for future reference or share them with others.

4. **Use the Dice Roll Tool:**
   - The app includes a dice roll tool to add an element of chance to your game. Use the `/roll` command to roll a dice with a specified number of sides (default is 100).

5. **Manage Knowledge:**
   - The app can load and use knowledge documents to enrich the narrative. These documents can provide background information, lore, and additional context to your story.

### Features

- **Flexible Tone and Content:**
  - The GM can shift the tone and content based on your cues. Whether you want a comedic, romantic, grim, or intense story, the GM will adapt accordingly.
  - The GM remains ethically neutral, respecting your boundaries and preferences.

- **Rich Descriptions and World-Building:**
  - The GM paints vivid environments, distinct cultures, and believable characters, balancing detail with pacing to keep the story engaging.

- **Adaptation and Improvisation:**
  - The GM uses "Yes, and…" or "Yes, but…" techniques to incorporate new ideas and evolve the plot organically from your actions.

- **Clear Decision Points:**
  - The GM presents meaningful choices that impact the story's direction, allowing you to shape the narrative.

- **Out-of-Character (OOC) Notes:**
  - System notes, dice rolls, and mechanic clarifications are placed in square brackets [like this] for clarity.

- **Neutral Ethical Stance:**
  - The GM can handle themes that are either cozy or intense, responding to your direction without imposing bias.

### Example Scenarios

- **Setting a Scene:**
  - **GM:** The wooden floor creaks beneath your boots as you step into a dimly lit cabin. A single oil lamp illuminates frayed maps pinned to the walls, revealing an ominous coastline drawn in red ink.
  - **Player:** I examine the maps to see if I recognize any landmarks.
  - **GM:** They depict jagged cliffs and scattered islands. If you lean closer, you might notice faint notes scrawled along the margins, mentioning hidden passages. The question is: do you want to investigate the next location at night or rest until morning?

- **Multi-NPC Social Encounter:**
  - **GM:** In the bustling tavern, three notable patrons gather. A cloaked wanderer sips quietly by the fireplace, a traveling merchant waves a colorful scarf for sale, and the local bard strums a lute near the bar.
  - **Player:** I approach the wanderer. Maybe they’ve heard rumors about the artifact I seek.
  - **GM:** Wanderer (voice quiet): “Depends who’s asking. The roads are dangerous for those chasing old legends.”  
  - **Merchant (calling out):** “Ah, but coin can buy safety, friend! Care for a scarf or two?”  
  - **Bard (hums a tune):** “Or you can pay me to compose a mighty ballad of your quest.”

- **Dramatic Combat:**
  - **GM:** Two armed bandits block the forest trail ahead, each brandishing a crude blade. One yells, “Hand over your valuables!”
  - **Player:** I step back, draw my bow, and warn them to leave me alone.
  - **GM:** The lead bandit smirks. “Got some nerve, do you?” He signals the other to flank your left. [OOC: Roll for initiative or declare another action. If you want to intimidate them, we can resolve that too.]

- **Puzzle or Riddle:**
  - **GM:** Deep in the catacombs, a sealed gate bears a riddle etched in marble: “When day merges with night, truth awakens in the silver light.” A circular depression suggests a missing object or key.
  - **Player:** I investigate the altar nearby for anything that might fit that depression.
  - **GM:** The altar holds a sun-and-moon motif carved in stone. If you’d like to attempt a lore check or solve the riddle directly, feel free.

### Local Storage

All your stories, characters, and elements are stored locally in a PostgreSQL database. This ensures that your data is safe and accessible, and you can resume your adventures at any time.

### Support and Feedback

If you have any questions, suggestions, or encounter any issues, feel free to open an issue in the repository or reach out to the community. We're here to help you create the best possible narrative experiences!

### Contributing

Feel free to contribute to this project by opening issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
