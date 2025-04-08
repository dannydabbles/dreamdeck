# Dreamdeck Improvement Plan

This plan outlines the steps to enhance the Dreamdeck application by integrating Chainlit features more deeply,
improving configuration management, and adding new functionalities like message deletion.

**Understanding & Context:**

1.  **Current State:** The app uses LangGraph for workflow management (`chat_workflow` in `src/workflows.py`), routing user input via a `decision_agent` to specialized agents (`writer_agent`, `dice_agent`, `web_search_agent`, `storyboard_editor_agent`, `todo_agent`). State is managed in `ChatState` (Pydantic model) and persisted partly in Chainlit's history and a ChromaDB vector store (`VectorStore` in `src/stores.py`). Configuration is handled via `config.yaml` and `src/config.py`. Chainlit integration happens in `src/event_handlers.py`. Prompts are currently embedded in `config.yaml`. Image generation is handled separately. Testing uses helper functions (`_dice_roll`, `_generate_story`, etc.) due to context issues.
2.  **Chainlit Concepts:**
    *   **`@cl.command`:** Allows defining slash commands (e.g., `/roll`, `/todo`) callable directly from the chat input. Useful for bypassing the decision agent.
    *   **`cl.ChatSettings`:** Defines user-configurable settings (sliders, text inputs) accessible via the settings icon in the chat UI. Perfect for overriding agent parameters like temperature or endpoints. Defined in `on_chat_start`.
    *   **`cl.Action`:** Buttons attached to messages, triggering callbacks (`@cl.action_callback`). Ideal for the "delete message" functionality.
    *   **`cl.Message`:** Represents messages in the UI. Has `id`, `parent_id`, `actions`, `elements`. Can be created, sent, and potentially updated or deleted (need to verify deletion API in v2).
    *   **`cl.user_session`:** A dictionary for storing session-specific data, like our `ChatState`, `VectorStore` instance, and user overrides from `cl.ChatSettings`.
    *   **`on_chat_start`, `on_message`, `on_chat_resume`:** Key lifecycle hooks for initialization, message handling, and resuming sessions.
    *   **`Elements`:** Used for displaying non-text content like images (already used for storyboards).
3.  **Integration Strategy:**
    *   Use `@cl.command` for direct agent access.
    *   Use `cl.ChatSettings` in `on_chat_start` to define user overrides for agent parameters. Agents will need to read these from `cl.user_session` and merge with defaults from `config.yaml`.
    *   Use `cl.Action` on messages to trigger a deletion callback.
    *   The deletion callback (`@cl.action_callback`) needs to:
        *   Modify the `ChatState` in `cl.user_session` (remove the message).
        *   Update the `VectorStore` (remove the corresponding entry).
        *   Remove the message from the Chainlit UI (if possible).
    *   Refactor config to separate prompts and add per-agent endpoint/parameter overrides.
    *   Ensure agent/workflow structure remains modular.

**Improvement Plan:**

**Phase 1: Configuration Refactoring & Prompt Externalization**

*   **Goal:** Clean up `config.yaml`, make prompts external, and prepare agent configurations for overrides.
*   **Tasks:**
    1.  Create a new directory `src/prompts/`.
    2.  ✅ Move the Jinja template content for `web_search_prompt`, `dice_processing_prompt`, `ai_writer_prompt`, `storyboard_generation_prompt`, and `decision_prompt` from `config.yaml` into separate files within `src/prompts/` (e.g., `web_search.j2`, `dice_processing.j2`, etc.).
    3.  Update `config.yaml`:
        *   ✅ Replace the prompt content under `prompts:` with filenames (e.g., `web_search_prompt: "web_search.j2"`).
        *   ✅ Modify the `agents:` section. For each agent (`decision_agent`, `writer_agent`, etc.), add optional fields for `base_url`. Define the *current* values as defaults here.
        ```yaml
        # Example for writer_agent in config.yaml
        agents:
          # ... other agents
          writer_agent:
            temperature: 0.7
            max_tokens: 8000
            streaming: true
            verbose: true
            # Add optional overrides - these are defaults if not set by user
            base_url: "http://192.168.1.111:5000/v1" # Default endpoint
            # temperature_override: null # Can be set via ChatSettings
            # max_tokens_override: null # Can be set via ChatSettings
        ```
    4.  Update `src/config.py`:
        *   Modify the Pydantic models (`DecisionAgentConfig`, `WriterAgentConfig`, etc.) to include the new optional fields (`base_url: Optional[str] = None`, etc.).
        *   ✅ Add fields to `ConfigSchema` to store the *loaded prompt content* (not just filenames).
        *   ✅ Add logic during config loading to read the prompt files specified in `config.yaml` (using the filenames) and store their content in the `config` object.
        ```python
        # Example in src/config.py (Actual implementation done)
        from pathlib import Path
        # ... other imports

        PROMPTS_DIR = Path(__file__).parent / "prompts"

        class ConfigSchema(BaseModel):
            # ... existing fields ...
            prompt_files: dict = Field(alias="prompts") # Store filenames from yaml
            loaded_prompts: dict = {} # Store loaded content

        def load_config():
            config_data = yaml.safe_load(CONFIG_FILE.read_text())
            schema = ConfigSchema.model_validate(config_data)

            # Load prompts from files
            for key, filename in schema.prompt_files.items():
                prompt_path = PROMPTS_DIR / filename
                if prompt_path.exists():
                    schema.loaded_prompts[key] = prompt_path.read_text()
                else:
                    cl_logger.error(f"Prompt file not found: {prompt_path}")
                    schema.loaded_prompts[key] = f"ERROR: Prompt file {filename} not found."
            return schema

        config = load_config() # Actual implementation done

        # ✅ Update constants to use loaded prompts
        AI_WRITER_PROMPT = config.loaded_prompts.get("ai_writer_prompt", "Write a story.")
        # ... other prompts ...

        # ✅ Update agent configs to expose new fields if needed
        # Example:
        WRITER_AGENT_BASE_URL = config.agents.writer_agent.base_url
        # ... other agent base URLs ...
        ```
    5.  Update agent files (`src/agents/*.py`): Modify the `_helper` functions (e.g., `_generate_story`, `_dice_roll`) to use the loaded prompt content from `src.config` (e.g., `src.config.AI_WRITER_PROMPT`) instead of accessing `config.prompts`. Ensure they use the base URL from `src.config` when initializing `ChatOpenAI`.

*   **Chainlit Docs Snippet:** (Not directly applicable, but sets up for Phase 3)

**Phase 2: Implement Chainlit Commands**

*   **Goal:** Allow direct invocation of agents via slash commands.
*   **Tasks:**
    1.  ✅ Create a new file `src/commands.py`.
    2.  ✅ In `src/commands.py`, define async functions decorated with `@cl.command` for each agent: `roll`, `search`, `todo`, `write`, `storyboard`.
    3.  ✅ Each command function should:
        *   ✅ Accept user input (e.g., `query: str`).
        *   ✅ Retrieve the current `ChatState` and `VectorStore` from `cl.user_session`.
        *   ✅ Create a `HumanMessage` from the `query`.
        *   ✅ *Option B (More integrated):* Add the `HumanMessage` to the main `ChatState`, call the agent *task* (e.g., `dice_agent(state)`), add the agent's `AIMessage` response back to the `ChatState`, update the vector store, and send the `cl.Message`. This keeps the command interaction in the history.
    4.  ✅ In `src/event_handlers.py`, import the command functions from `src/commands.py` to make them available to Chainlit.

*   **Chainlit Docs Snippet:**
    > ```python
    > @cl.command(name="my-command", description="Description of my command")
    > async def my_command(query: str):
    >     # query is the text entered by the user after the command name
    >     await cl.Message(content=f"Command received: {query}").send()
    > ```

**Phase 3: Implement Chainlit Chat Settings**

*   **Goal:** Allow users to override agent parameters via the UI.
*   **Tasks:**
    1.  In `src/event_handlers.py` (`on_chat_start`):
        *   Define `cl.ChatSettings`.
        *   Add input fields (`cl.input_widget.TextInput`, `cl.input_widget.Slider`) for parameters you want to make configurable (e.g., `Writer Temperature`, `Writer Max Tokens`, `Writer Endpoint`, `Storyboard Endpoint`, `Decision Temp`, etc.). Use descriptive IDs.
        *   Load default values from `src.config` (e.g., `src.config.WRITER_AGENT_TEMPERATURE`).
        ```python
        # src/event_handlers.py
        import chainlit as cl
        from chainlit.input_widget import Slider, TextInput # Import widgets
        from src import config # Import your config

        @cl.on_chat_start
        async def on_chat_start():
            # ... existing setup ...

            settings = await cl.ChatSettings(
                [
                    Slider(
                        id="writer_temp",
                        label="Writer Agent - Temperature",
                        min=0.0, max=2.0, step=0.1, initial=config.WRITER_AGENT_TEMPERATURE
                    ),
                    TextInput(
                        id="writer_endpoint",
                        label="Writer Agent - OpenAI Endpoint URL",
                        initial=config.WRITER_AGENT_BASE_URL or "", # Use loaded default
                        placeholder="e.g., http://localhost:5000/v1"
                    ),
                    # ... Add settings for other agents/parameters ...
                    Slider(
                        id="storyboard_temp",
                        label="Storyboard Agent - Temperature",
                        min=0.0, max=2.0, step=0.1, initial=config.STORYBOARD_EDITOR_AGENT_TEMPERATURE
                    ),
                     TextInput(
                        id="storyboard_endpoint",
                        label="Storyboard Agent - OpenAI Endpoint URL",
                        initial=config.STORYBOARD_EDITOR_AGENT_BASE_URL or "", # Assuming you add this to config
                        placeholder="e.g., http://localhost:5000/v1"
                    ),
                ]
            ).send() # Send settings to the UI

            # Store initial settings in the session if needed, or retrieve on demand
            # cl.user_session.set("chat_settings", settings) # Settings are automatically available

            # ... rest of on_chat_start ...
        ```
    2.  Modify agent `_helper` functions (e.g., `_generate_story`, `_generate_storyboard`) in `src/agents/*.py`:
        *   Before initializing `ChatOpenAI`:
            *   Retrieve user settings using `cl.user_session.get("chat_settings")`. This returns a dictionary like `{"writer_temp": 0.8, "writer_endpoint": "..."}`.
            *   Get the default values from `src.config`.
            *   Determine the final value to use, prioritizing the user setting if it exists and is valid, otherwise falling back to the default.
            *   Pass the final values (`temperature`, `base_url`, `max_tokens`) to the `ChatOpenAI` constructor.
        ```python
        # Example in src/agents/writer_agent.py (_generate_story)
        from langchain_openai import ChatOpenAI
        from src import config
        import chainlit as cl

        async def _generate_story(state: ChatState) -> list[BaseMessage]:
            try:
                # ... get prompt, etc. ...

                # Get user settings and defaults
                user_settings = cl.user_session.get("chat_settings", {})
                final_temp = user_settings.get("writer_temp", config.WRITER_AGENT_TEMPERATURE)
                final_endpoint = user_settings.get("writer_endpoint") or config.WRITER_AGENT_BASE_URL # Prioritize non-empty user setting
                # final_max_tokens = user_settings.get("writer_max_tokens", config.WRITER_AGENT_MAX_TOKENS) # If you add this setting

                # Initialize the LLM with potentially overridden settings
                llm = ChatOpenAI(
                    base_url=final_endpoint, # Use final endpoint
                    temperature=final_temp, # Use final temperature
                    max_tokens=config.WRITER_AGENT_MAX_TOKENS, # Use default or add override
                    streaming=config.WRITER_AGENT_STREAMING,
                    verbose=config.WRITER_AGENT_VERBOSE,
                    timeout=config.LLM_TIMEOUT,
                )

                # ... rest of the function ...
            except Exception as e:
                # ... error handling ...
        ```
*   **Chainlit Docs Snippet:**
    > ```python
    > from chainlit.input_widget import Select, Slider, Switch
    >
    > settings = await cl.ChatSettings(
    >     [
    >         Select(
    >             id="Model",
    >             label="OpenAI - Model",
    >             values=["gpt-3.5-turbo", "gpt-4"],
    >             initial_index=0,
    >         ),
    >         Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
    >         Slider(
    >             id="Temperature",
    >             label="OpenAI - Temperature",
    >             min=0.0,
    >             max=1.0,
    >             step=0.1,
    >             initial=1.0,
    >         ),
    >     ]
    > ).send()
    > # settings is a dict {'Model': 'gpt-3.5-turbo', 'Streaming': True, 'Temperature': 1.0}
    > ```

**Phase 4: Message Deletion - Backend Logic**

*   **Goal:** Implement the logic to remove messages from `ChatState` and `VectorStore`.
*   **Tasks:**
    1.  Modify `src/stores.py` (`VectorStore`):
        *   Update the `put` method to accept an optional `message_id: str` and store it in the ChromaDB document
metadata.
        *   Add a `delete_by_message_id(self, message_id: str)` method that queries ChromaDB for documents with the matching `message_id` in their metadata and deletes them. ChromaDB's `delete` method usually works with document IDs, so you might need to `get` the document first to find its internal Chroma ID based on the metadata query, then `delete` using that ID.
        ```python
        # src/stores.py
        # ... imports ...
        from chromadb.types import Where # Import Where

        class VectorStore:
            # ... __init__ ...

            async def put(self, content: str, message_id: Optional[str] = None) -> None:
                """Store new content in ChromaDB with optional message_id metadata."""
                doc_id = str(uuid.uuid4())
                metadata = {"message_id": message_id} if message_id else None
                await asyncio.to_thread(
                    self.collection.add,
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[metadata] if metadata else None # Add metadata if provided
                )

            async def delete_by_message_id(self, message_id: str) -> None:
                """Delete documents from ChromaDB based on message_id metadata."""
                if not message_id:
                    return
                try:
                    # Query to find ChromaDB internal IDs based on metadata
                    results = await asyncio.to_thread(
                        self.collection.get,
                        where=Where({"message_id": message_id}) # Use Where clause
                    )
                    ids_to_delete = results.get("ids", [])
                    if ids_to_delete:
                        cl_logger.info(f"Deleting vector store entries for message_id: {message_id}, Chroma IDs: {ids_to_delete}")
                        await asyncio.to_thread(
                            self.collection.delete,
                            ids=ids_to_delete
                        )
                    else:
                         cl_logger.warning(f"No vector store entries found for message_id: {message_id}")
                except Exception as e:
                    cl_logger.error(f"Failed to delete vector store entry for message_id {message_id}: {e}")

            # ... get, add_documents ...
        ```
    2.  Modify `src/models.py` (`ChatState`):
        *   Add a `delete_message(self, message_id: str, vector_store: VectorStore)` method.
        *   This method should:
            *   Find the index of the message in `self.messages` where `message.metadata.get('message_id') == message_id`.
            *   If found, store the message content.
            *   Remove the message from `self.messages`.
            *   Call `await vector_store.delete_by_message_id(message_id)`.
            *   *Note:* Handling child messages automatically is complex with the current flat list structure. We'll start by deleting only the target message. Recursive deletion would require traversing `parent_id` relationships in the Chainlit UI messages, which isn't directly represented in our `ChatState` list.
        ```python
        # src/models.py
        # ... imports ...
        from src.stores import VectorStore # Import VectorStore

        class ChatState(BaseModel):
            # ... existing fields and methods ...

            async def delete_message(self, message_id: str, vector_store: VectorStore):
                """Deletes a message from the state and vector store by its Chainlit ID."""
                message_to_delete = None
                message_index = -1

                for i, msg in enumerate(self.messages):
                    # Ensure metadata exists and contains the key before accessing
                    if isinstance(msg, (HumanMessage, AIMessage)) and hasattr(msg, 'metadata') and msg.metadata and msg.metadata.get("message_id") == message_id:
                        message_to_delete = msg
                        message_index = i
                        break

                if message_index != -1:
                    cl_logger.info(f"Deleting message with ID: {message_id} from ChatState.")
                    del self.messages[message_index]
                    # Delete from vector store
                    await vector_store.delete_by_message_id(message_id)
                else:
                    cl_logger.warning(f"Message with ID: {message_id} not found in ChatState for deletion.")

            # Add metadata field to BaseMessage if needed, or rely on dynamic addition
            # Pydantic v2 allows extra fields if not forbidden, or use metadata dict
            # Ensure messages have metadata: dict = {}
            # Langchain BaseMessage already has a metadata field
    ```
    3.  Modify message creation points (`on_message`, agent responses that add to state) to include the `cl.Message.id` in the `metadata` of the `HumanMessage` or `AIMessage` being added to `ChatState`. This requires getting the ID *after* the `cl.Message` is sent.
        ```python
        # Example modification in on_message after invoking workflow
        # This is conceptual - the workflow needs to return the cl.Message objects or their IDs

        # --- Inside on_message (Conceptual - needs adjustment based on workflow return) ---
        # Assume workflow now somehow provides the ID of the cl.Message it sent
        # Maybe the agent sends the message and returns its ID?

        # --- Modification within an agent that sends a message ---
        # Example in src/agents/dice_agent.py (_dice_roll)
        async def _dice_roll(state: ChatState) -> List[BaseMessage]:
            # ... logic ...
            try:
                # ... generate results ...
                cl_msg_content = f"**Dice Rolls:**\n\n..."
                cl_msg = CLMessage(content=cl_msg_content, parent_id=None)
                await cl_msg.send() # Send the message

                lang_graph_msg_content = "\n".join([...])
                # Create AIMessage with metadata containing the ID
                ai_msg = AIMessage(
                    content=lang_graph_msg_content,
                    name="dice_roll",
                    metadata={"message_id": cl_msg.id} # Store the ID here
                )
                return [ai_msg]
            except Exception as e:
                # ... error handling ...

        # --- Modification in on_message to store user message ID ---
        @cl.on_message
        async def on_message(message: cl.Message):
            # ... get state, vector_memory ...
            if message.type != "user_message": return

            try:
                # Add user message to state WITH metadata
                user_msg = HumanMessage(
                    content=message.content,
                    name="Player",
                    metadata={"message_id": message.id} # Store user message ID
                )
                state.messages.append(user_msg)
                await vector_memory.put(content=message.content, message_id=message.id) # Pass ID to put

                # ... rest of the logic ...

                # After workflow runs, ensure the AI message added to state also has its ID
                # This requires the workflow/agents to return messages with metadata populated
                # Example: last_ai_msg = state.messages[-1]
                # if isinstance(last_ai_msg, AIMessage) and "message_id" in last_ai_msg.metadata:
                #    await vector_memory.put(content=last_ai_msg.content, message_id=last_ai_msg.metadata["message_id"])


            except Exception as e:
                # ... error handling ...
        ```

*   **Chainlit Docs Snippet:** (Focus on `cl.Message` properties and `cl.user_session`)

**Phase 5: Message Deletion - UI Integration**

*   **Goal:** Add "Delete" buttons to messages and handle the callback.
*   **Tasks:**
    1.  Define the action callback in `src/event_handlers.py`:
        ```python
        # src/event_handlers.py
        import chainlit as cl
        from src.models import ChatState
        from src.stores import VectorStore

        @cl.action_callback("delete_message")
        async def on_delete_message(action: cl.Action):
            message_id = action.value # Get message ID from action value
            if not message_id:
                await cl.ErrorMessage(content="Error: Delete action missing message ID.").send()
                return

            await cl.Message(content=f"Attempting to delete message {message_id}...").send() # User feedback

            state: ChatState = cl.user_session.get("state")
            vector_store: VectorStore = cl.user_session.get("vector_memory")

            if not state or not vector_store:
                await cl.ErrorMessage(content="Error: Session state not found for deletion.").send()
                return

            try:
                # Call the backend deletion logic
                await state.delete_message(message_id, vector_store)

                # Update state in session
                cl.user_session.set("state", state)

                # Remove message from UI (Requires Chainlit API call - check docs for v2)
                # Placeholder - Verify correct API call for Chainlit v2
                try:
                     # Check Chainlit documentation for the correct way to remove a message by ID in v2.x
                     # This might involve sending a specific message type or using a JS call.
                     # For now, log and inform user.
                     cl_logger.info(f"Message {message_id} deleted from state/vector store. UI removal needs verification.")
                     await cl.Message(content=f"Message {message_id} deleted from history. UI may need refresh to reflect removal.").send()
                except Exception as ui_err:
                     cl_logger.error(f"Error attempting UI removal for message {message_id}: {ui_err}")
                     await cl.ErrorMessage(content=f"Failed to update UI for message {message_id} removal.").send()

            except Exception as e:
                cl_logger.error(f"Failed to delete message {message_id}: {e}")
                await cl.ErrorMessage(content=f"Error deleting message {message_id}.").send()
        ```
    2.  Modify message sending points (`on_chat_start`, agent `cl.Message.send()` calls, command responses) to add the `cl.Action`. This is tricky because the `message.id` is usually assigned *after* sending. The best approach is often to send, then update.
        ```python
        # Example modification for an agent sending a message
        # In src/agents/dice_agent.py (_dice_roll)

        # ... inside the try block ...
        cl_msg_content = f"**Dice Rolls:**\n\n..."
        cl_msg = CLMessage(content=cl_msg_content, author="Dice Roller") # Set author clearly
        await cl_msg.send() # Send first to get the ID

        # Now update the message to add the action
        delete_action = cl.Action(
            name="delete_message",
            value=cl_msg.id, # Use the ID from the sent message
            label="Delete",
            description="Remove this dice roll message"
        )
        cl_msg.actions = [delete_action]
        await cl_msg.update() # Update the message in the UI

        # Create AIMessage for state with metadata
        ai_msg = AIMessage(
            content=lang_graph_msg_content,
            name="dice_roll",
            metadata={"message_id": cl_msg.id} # Store the ID
        )
        return [ai_msg]

        # --- Similar update needed for user messages in on_message ---
        # This is harder as on_message receives the message object directly.
        # Chainlit v2 might allow adding actions to incoming messages via config or hooks.
        # If not, we might need to intercept/wrap the message display or use JS.
        # Let's focus on agent messages first. We can add actions to the initial GM message too.
        # In on_chat_start:
        initial_msg = await cl.Message(content=START_MESSAGE, author="Game Master").send()
        initial_msg.actions = [cl.Action(name="delete_message", value=initial_msg.id, label="Delete")]
        await initial_msg.update()
        # Ensure the AIMessage in initial state also gets the ID in metadata
        state = ChatState(
            messages=[AIMessage(content=START_MESSAGE, name="Game Master", metadata={"message_id": initial_msg.id})],
            # ... rest of state init ...
        )

        ```
*   **Chainlit Docs Snippet:**
    > ```python
    > actions = [
    >     cl.Action(name="action_button", value="example_value", description="Click me!")
    > ]
    >
    > await cl.Message(content="Here is a message with an action", actions=actions).send()
    >
    > @cl.action_callback("action_button")
    > async def on_action(action: cl.Action):
    >     print("The user clicked on the action button!")
    >     # Action value is available in action.value
    >     await cl.Message(content=f"Action {action.name} received with value {action.value}").send()
    >     # Optionally remove the action button from the message
    >     await action.remove()
    > ```
    > (Note: `action.remove()` removes the button, not the message. Message removal API needs verification for v2).

**Phase 6: Structure for Expansion & Bug Fixes**

*   **Goal:** Ensure the codebase is robust, maintainable, and ready for future agents/workflows. Fix any minor
issues found.
*   **Tasks:**
    1.  **Review Agent/Workflow Structure:** Confirm the pattern of `@task` calling `_helper` is consistently applied. Ensure `ChatState` provides sufficient context. The current `chat_workflow` using `decision_agent` is suitable for adding more *tools* or *agents* triggered by the decision. Adding entirely new *workflows* (e.g., a document processing workflow) might involve creating separate LangGraph graphs or entry points, potentially selectable via Chat Profiles or initial user choice.
    2.  **Review Config:** Check `config.yaml` and `src/config.py` for clarity and ease of adding new agent sections or global settings.
    3.  **Bug Fixes:** Address any small bugs or inconsistencies noticed during previous phases (e.g., logging, error handling, prompt formatting issues, ensuring `message_id` is consistently added to metadata).
    4.  **Testing:** Update existing tests to reflect changes (especially config loading and agent initialization). Add new tests for commands and deletion logic *if feasible* within the context limitations (testing Chainlit callbacks and UI interactions might be hard). Continue using the `_helper` pattern for direct agent logic tests.
    5.  **Code Quality:** Run linters/formatters (`make format`, `make lint`). Add type hints where missing. Review error handling and logging.

