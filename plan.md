# Dreamdeck Improvement Plan (Revised)

This plan outlines the next phases of development for the Dreamdeck application, building upon the initial setup and focusing on enhancing player experience, stability, and core storytelling features.

**Understanding & Context:**

1.  **Current State:** The app uses LangGraph for workflow management (`chat_workflow` in `src/workflows.py`), routing user input via a `decision_agent` or direct `@cl.command` calls to specialized agents (`writer_agent`, `dice_agent`, `web_search_agent`, `storyboard_editor_agent`, `todo_agent`). State is managed in `ChatState` (Pydantic model) and persisted partly in Chainlit's history and a ChromaDB vector store (`VectorStore` in `src/stores.py`). Configuration (`config.yaml`, `src/config.py`) includes externalized prompts and agent parameter defaults. Chainlit `ChatSettings` allow user overrides for agent parameters (temp, max_tokens, endpoint). Chainlit integration happens in `src/event_handlers.py`.
2.  **Completed Work (Original Plan):**
    *   Phase 1: Configuration Refactoring & Prompt Externalization.
    *   Phase 2: Implemented Chainlit Commands (`/roll`, `/search`, etc.).
    *   Phase 3: Implemented Chainlit Chat Settings for agent parameter overrides.
3.  **Chainlit Concepts:** We leverage `@cl.command`, `cl.ChatSettings`, `cl.Message`, `cl.user_session`, lifecycle hooks (`on_chat_start`, `on_message`, etc.), and `Elements` (for images). We can further explore `cl.Action`, `cl.AskUser`, `cl.Step`, and `Chat Profiles`.
4.  **Revised Strategy:** Focus on stabilizing the current features, enhancing player interaction through explicit choices, deepening the narrative state, improving observability, and then considering advanced features like message editing or more complex state management. Full message deletion is deferred due to complexity and potential impact on narrative consistency.

**Improvement Plan:**

**Phase 1: Stabilization & Core Refinement**

*   **Goal:** Solidify the existing features, improve observability, ensure configuration is comprehensive, and fix bugs.
*   **Tasks:**
    1.  **Bug Fixing & Polish:**
        *   Address any known bugs or inconsistencies from the initial implementation (logging, error handling, UI glitches).
        *   Ensure `message_id` is consistently captured in `AIMessage` and `HumanMessage` metadata within `ChatState` for all creation points (commands, `on_message`, agent responses).
        *   Verify that `VectorStore.put` is always called with the `message_id` when available.
    2.  **Enhance Observability with `cl.Step`:**
        *   In `src/workflows.py` (`_chat_workflow`) and potentially within key agent functions (`_generate_story`, `_dice_roll`, etc.), wrap logical blocks or agent calls with `async with cl.Step(name="...", type="...") as step:` (e.g., "Deciding Action", "Rolling Dice", "Generating Story Segment", "Invoking Web Search").
        *   Set `step.input` and `step.output` where appropriate to show data flow (e.g., input prompt, agent decision, tool results, generated text). This will help debug the flow directly in the UI.
        ```python
        # Example in _chat_workflow
        async with cl.Step(name="Decide Action", type="llm") as step:
            decision_response = await decision_agent(state)
            action = decision_response[0].name
            step.output = action # Log the decided action
            # ... rest of decision logic ...

        # Example in _dice_roll
        async with cl.Step(name="Process Dice Request", type="tool") as step:
            step.input = input_msg.content
            # ... llm call for specs/reasons ...
            # ... perform rolls ...
            step.output = lang_graph_msg # Log the formatted result
            # ... send cl.Message ...
        ```
    3.  **Comprehensive Chat Settings:**
        *   Review `src/event_handlers.py` (`on_chat_start`) and `src/config.py`. Ensure all key agent parameters (temperature, max_tokens, base_url) for *all* agents (`decision`, `writer`, `storyboard`, `dice` LLM, `web_search` LLM) are exposed via `cl.ChatSettings`.
        *   Ensure all agents correctly read and prioritize these user settings over the defaults from `config.yaml`.
    4.  **Storyboard Reliability:**
        *   Review the `generate_storyboard` task and `process_storyboard_images` in `src/agents/storyboard_editor_agent.py`. Improve error handling and logging.
        *   Ensure the `gm_message_id` is reliably passed and used for parenting the image messages.
    5.  **Testing:**
        *   Update existing tests (`tests/`) to reflect the use of `cl.Step` if it impacts logic flow (it shouldn't significantly).
        *   Add tests specifically verifying that agents correctly pick up overridden parameters from mocked `cl.user_session.get("chat_settings")`.

**Phase 2: Enhancing Player Interaction & State**

*   **Goal:** Introduce explicit player choices and begin tracking basic character/world state.
*   **Tasks:**
    1.  **Explicit Choices with `cl.Action`:**
        *   Modify the `writer_agent` (`_generate_story` in `src/agents/writer_agent.py`). When the GM should present distinct choices (as described in the `AI_WRITER_PROMPT` guidelines), the LLM response should include a marker or structured format indicating these choices (e.g., `[CHOICE: Go left | Go right | Wait here]`).
        *   Update the logic that processes the `writer_agent` response (likely in `_chat_workflow` or just before sending the `cl.Message`). If choices are detected:
            *   Parse the choices.
            *   Create `cl.Action` buttons for each choice. The `action.value` should contain the text of the choice. Use a unique `action.name` (e.g., "make_choice").
            *   Attach these actions to the `cl.Message` sent by the GM.
        *   Implement an `@cl.action_callback("make_choice")` in `src/event_handlers.py`. This callback should:
            *   Get the chosen text from `action.value`.
            *   Create a `HumanMessage` representing the player's choice (e.g., `HumanMessage(content=action.value)`).
            *   Add this message to the `ChatState`.
            *   Optionally, disable or remove the choice buttons (`await action.remove()` might remove just the clicked one, or update the message to remove all).
            *   Trigger the `_chat_workflow` again to continue the story based on the choice.
    2.  **Basic Character State:**
        *   Modify `src/models.py`: Add a simple dictionary `character_state: dict = {}` to the `ChatState` model.
        *   Modify the `AI_WRITER_PROMPT`: Instruct the GM to occasionally update or reference simple character state elements (like inventory items or key conditions) using a specific format (e.g., `[STATE_UPDATE: inventory={'key': 1}, status='Injured']`).
        *   In `src/workflows.py` (`_chat_workflow`), after the `writer_agent` responds:
            *   Check the AI message content for the state update format.
            *   If found, parse the update and merge it into `state.character_state`.
            *   Ensure the updated `state` (including `character_state`) is saved back to `cl.user_session`.
        *   Modify `AI_WRITER_PROMPT` again: Instruct the GM to consider the `{{ state.character_state }}` (pass this into the template render context) when generating the narrative.
        *   (Optional) Display Character State: Add a small section in the UI (perhaps using `cl.Text` element updated periodically or on demand via a command) to show the current `character_state`.

**Phase 3: Storytelling & World Building Enhancements**

*   **Goal:** Improve the storyboard feature and potentially add basic NPC tracking.
*   **Tasks:**
    1.  **Storyboard Control:**
        *   Add `ChatSettings` for storyboard parameters (e.g., aspect ratio preset, maybe a style keyword input).
        *   Modify `src/image_generation.py` and `src/agents/storyboard_editor_agent.py` to use these settings when generating prompts and calling the image API.
        *   Consider adding a `cl.Action` to GM messages: "Regenerate Storyboard". The callback would re-run `storyboard_editor_agent` for that specific GM message ID.
    2.  **Basic NPC Tracking:**
        *   Similar to character state, add `npc_state: dict = {}` to `ChatState`.
        *   Update `AI_WRITER_PROMPT` to instruct the GM to track key NPCs encountered and their basic status or relationship to the player using a format like `[NPC_UPDATE: 'Guard Captain': {'status': 'Suspicious', 'location': 'Gate'}, 'Mystic': {'status': 'Helpful'}]`.
        *   Update `_chat_workflow` to parse and merge these updates into `state.npc_state`.
        *   Pass `{{ state.npc_state }}` into the `AI_WRITER_PROMPT` context for the GM to reference.

**Phase 4: Advanced Features & Polish**

*   **Goal:** Explore message editing, chat profiles, and further memory enhancements.
*   **Tasks:**
    1.  **Message Editing:**
        *   Investigate Chainlit v2 capabilities for message editing. Can `cl.Message.update()` change content effectively?
        *   If feasible, add an "Edit" `cl.Action` to user messages.
        *   The callback (`@cl.action_callback("edit_message")`) would likely need to:
            *   Use `cl.AskUserMessage` to get the new content from the user.
            *   Update the corresponding message in `ChatState`.
            *   Update the entry in `VectorStore` (delete old, put new).
            *   Update the message in the Chainlit UI using `cl.Message(id=...).update(content=...)`.
            *   *Challenge:* Decide if/how this should affect subsequent messages or trigger recalculations. Start simple (just edit the text).
    2.  **Chat Profiles:**
        *   Explore `cl.ChatProfile` to allow users to select different GM personas or game genres at the start of a chat.
        *   Each profile could load slightly different system prompts (especially for `writer_agent`) or potentially different default `ChatSettings`.
        *   Implement different prompt files (e.g., `ai_writer_prompt_fantasy.j2`, `ai_writer_prompt_scifi.j2`) and load the appropriate one based on the selected profile in `on_chat_start`.
    3.  **Memory Enhancements:**
        *   Review the effectiveness of the current vector store memory (`state.memories`).
        *   Consider adding summarization steps or more sophisticated retrieval strategies if context seems lacking for the LLMs.

**Ongoing:**

*   **Testing:** Continuously update and add tests for new features, focusing on agent logic and state manipulation.
*   **Documentation:** Keep `README.md` and `chainlit.md` updated with new features and usage instructions. Add code comments.
*   **Code Quality:** Regularly run linters/formatters (`make format`, `make lint`). Refactor as needed for clarity and maintainability.
*   **Dependency Updates:** Periodically review and update dependencies (`poetry update`).

