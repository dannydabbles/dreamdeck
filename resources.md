Integration Toolbox for Interactive AI Applications
This document details how to integrate several powerful tools into a modular, scalable system for interactive applications. Although the examples are written in the context of storytelling, the techniques apply broadly. The key components include:

LangGraph Functional API – for building workflows as plain Python functions with automatic persistence and checkpointing.
Chainlit – for creating a rich, interactive chat UI that supports session, thread, and media persistence.
Pydantic – for defining and validating structured configuration and state data.
ChromaDB – for vector-based, long-term memory storage and semantic retrieval.
Pytest – for comprehensive testing of individual components and end-to-end workflows.
Below you’ll find detailed sections, code examples, and explanations to help you implement and extend these integrations.

1. Application Configuration & State Management with Pydantic
Using Pydantic not only validates your configuration but also documents expected data shapes. It is essential for both configuration and maintaining a consistent workflow state.

1.1 Defining Application Configuration
Declare your configuration with type annotations. You can even auto-load environment variables using Pydantic’s configuration options.

from pydantic import BaseModel, Field, ConfigDict
from typing import List

class AppConfig(BaseModel):
    gm_agent_name: str = "Brennan"
    pc_agent_names: List[str] = Field(default_factory=lambda: ["Player1"])
    storyboard_api_url: str
    postgres_dsn: str
    chroma_db_path: str
    config_file_path: str = "config.yaml"

    model_config = ConfigDict(env_prefix="APP_")  # Load environment variables prefixed with APP_
1.2 Loading Configuration from YAML
Load configuration from a YAML file for easy customization.

import yaml

def load_config(config_path: str) -> AppConfig:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return AppConfig(**config_data)
1.3 Defining Workflow State
Define your conversation or story state as a Pydantic model to enforce structure.

from pydantic import BaseModel

class StoryState(BaseModel):
    conversation: str = ""
    directive: str = ""
    last_input: str = ""
    relevant_lore: str = ""
Using such models ensures that every piece of state passed between tasks conforms to a known schema, which is critical when persisting state or debugging complex flows.

2. LangGraph Functional API – Modular Workflows and Multi-Agent Architectures
The LangGraph Functional API enables you to compose workflows as regular Python functions using decorators. This approach supports persistence (with checkpointing), asynchronous execution, and easy integration with other components. It also simplifies migration from older graph-based paradigms.

2.1 Basic Workflow with Persistence and Store Integration
Define your core workflow using @entrypoint to mark the main function and @task for sub-tasks. The following example demonstrates a chat workflow that persists conversation history and interacts with an external store for memory.

from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
import uuid

# Assume persistent_store is provided by our ChromaStore (see Section 4)
persistent_store = None  

@task
def call_llm(user_message: str, context: str) -> str:
    # Replace with your actual LLM call (e.g., ChatOpenAI with streaming enabled)
    prompt = f"{context}\nUser: {user_message}\nAssistant:"
    return f"(LLM response to '{user_message}' with context: '{context}')"

@entrypoint(checkpointer=MemorySaver(), store=persistent_store)
def chat_workflow(message: str, *, previous: str = None, store=None, config: dict = None):
    conversation = previous or ""
    conversation += f"\nUser: {message}"
    response = call_llm(message, conversation).result()
    conversation += f"\nAssistant: {response}"
    # Optionally, store new facts if the message includes "remember"
    if "remember" in message.lower():
        fact_text = message.split("remember", 1)[1].strip()
        store.put(("facts", config.get("user_id", "default")),
                  key=str(uuid.uuid4()),
                  value={"data": fact_text})
    return entrypoint.final(value=response, save=conversation)
Key Points:

Checkpointing: Using MemorySaver() allows automatic persistence and resume via the previous argument.
Store Integration: The workflow uses an external store (e.g., a ChromaDB-backed store) to persist long-term facts.
Asynchronous Execution: Tasks are executed asynchronously; calling .result() awaits their output.
2.2 Migrating from Older Paradigms
Instead of defining a fixed graph with nodes and edges, you now define plain Python functions. For example, where you once had a node to update global state, simply write a function that returns the updated state:

def update_state(state: dict, new_data: dict) -> dict:
    state.update(new_data)
    return state
This modular approach makes code easier to read, test, and maintain.

2.3 Hierarchical Multi-Agent Architectures
Building on modular workflows, you can implement hierarchical multi-agent systems. This design allows a high-level Director to manage subordinate agents such as a Game Master (GM) and NPC/PC agents.

2.3.1 Director and GM Agents
@task
def director_decision(user_input: str) -> str:
    # Generate a high-level narrative directive.
    return f"Plot: {user_input} leads to a new challenge."

@entrypoint
def director_agent(user_input: str, *, previous: dict = None) -> dict:
    state = previous or {}
    directive = director_decision(user_input).result()
    state["directive"] = directive
    gm_response = gm_agent(f"Directive: {directive}").result()
    state["gm_response"] = gm_response
    return entrypoint.final(value={"directive": directive, "gm_response": gm_response}, save=state)

@entrypoint
def gm_agent(prompt: str) -> str:
    narrative = call_llm(prompt, "").result()
    return narrative
2.3.2 NPC Team Agent (Hierarchical Extension)
@task
def npc_response(character: str, context: str) -> str:
    # Generate NPC dialogue based on context.
    return f"{character} says: I react to {context}."

@entrypoint
def npc_team(state: dict, *, previous: dict = None) -> dict:
    state = previous or state
    npc_line = npc_response("Goblin King", state.get("gm_response", "")).result()
    state["npc_dialogue"] = npc_line
    return entrypoint.final(value=state, save=state)
Integration Tip: Chain these agents (Director → GM → NPC Team) so that each layer refines the narrative. This hierarchy is extensible: you can later add a PC agent or a supervisor that oversees decision-making for tool usage.

2.4 Asynchronous and Streaming Workflows
LangGraph supports asynchronous execution and streaming outputs (e.g., token-by-token response streaming) to improve responsiveness.

@cl.on_message
async def handle_streaming_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id, "user_id": "user123"}}
    full_response = ""
    async for chunk, meta in chat_workflow.stream(message.content, config=config, stream_mode=["tokens"]):
        full_response += chunk
        await cl.Message(content=chunk, author="Game Master", stream=True).send()
    await cl.Message(content=full_response, author="Game Master").send()
This pattern provides a live “typing” effect while the workflow is still generating the final narrative.

3. Chainlit Integration – Building a Rich UI and Managing Sessions
Chainlit handles the chat UI, session management, and rich media. It’s essential to keep Chainlit’s state synchronized with your LangGraph workflows.

3.1 Basic Message Handling
import chainlit as cl

@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("thread_id", cl.user_session.get("id"))
    cl.user_session.set("message_count", 0)

@cl.on_message
async def handle_message(message: cl.Message):
    count = cl.user_session.get("message_count") or 0
    cl.user_session.set("message_count", count + 1)
    thread_id = cl.user_session.get("thread_id")
    user_input = message.content
    config = {"configurable": {"thread_id": thread_id, "user_id": "user123"}}
    response = chat_workflow.invoke(user_input, config=config)
    await cl.Message(content=response, author="Assistant").send()
3.2 Resuming Sessions and Rebuilding LangGraph State
When a user resumes a chat, ensure you reload both Chainlit and LangGraph state.

@cl.on_chat_resume
async def on_chat_resume(thread: cl.ThreadDict):
    thread_id = thread.get("id")
    cl.user_session.set("thread_id", thread_id)
    stored_memory = get_chroma_memory(thread_id)  # Custom function to retrieve stored Chroma memory
    cl.user_session.set("chroma_memory", stored_memory)
    print(f"Resumed thread {thread_id}")
Always pass the same thread_id in the config so that LangGraph’s checkpointer can automatically load prior state.

3.3 Advanced UI Elements and Interactions
Chainlit supports multimedia elements and interactive options to enrich user experience.

3.3.1 Sending Images, Audio, and Video
@cl.on_chat_start
async def start():
    # Sending an image
    image = cl.Image(path="./scene1.jpg", name="Scene 1", display="inline")
    await cl.Message(content="The story begins at dusk by a riverside.", elements=[image]).send()
    
    # Sending audio and video (advanced)
    audio = cl.Audio(path="./intro.mp3", name="Intro Audio")
    video = cl.Video(path="./trailer.mp4", name="Trailer")
    await cl.Message(content="Listen to the introduction and watch the trailer.", elements=[audio, video]).send()
3.3.2 Presenting Options and Handling User Actions
@cl.on_chat_start
async def start_options():
    actions = [
        cl.Action(name="go_left", label="Go Left", icon="arrow-left", payload={"direction": "left"}),
        cl.Action(name="go_right", label="Go Right", icon="arrow-right", payload={"direction": "right"})
    ]
    await cl.Message(content="You stand at a crossroads. What do you do?", actions=actions).send()

@cl.action_callback("go_left")
async def on_left(action: cl.Action):
    await cl.Message(content="You chose the left path...").send()
3.3.3 Overriding DOM and Custom Interactions
Chainlit supports injecting custom HTML/CSS/JS if needed for specific UI tweaks. This is done via its configuration, though the details vary by project. Use these features to fine-tune the display of messages, style of action buttons, or integration of third-party widgets.

4. ChromaDB Integration – Long-Term Memory and Semantic Retrieval
ChromaDB is used for persisting vectorized representations of textual data, making it ideal for long-term memory in interactive applications.

4.1 Initializing and Using a Custom ChromaStore
Implement a custom store by extending LangGraph’s BaseStore. This allows your workflows to seamlessly store and retrieve facts.

import chromadb
from chromadb.config import Settings
from langgraph.store.base import BaseStore, StoreData

def embed(text: str) -> list:
    # Dummy embedding function – replace with your embedding model.
    return [ord(c) for c in text[:10]]  # simplistic example

class ChromaStore(BaseStore):
    def __init__(self, collection):
        self.collection = collection

    def put(self, namespace: tuple, key: str, value: dict):
        uid = f"{namespace}-{key}"
        doc = value.get("data", "")
        embedding = embed(doc)
        self.collection.add(documents=[doc], embeddings=[embedding], ids=[uid])

    def get(self, namespace: tuple, key: str) -> StoreData:
        uid = f"{namespace}-{key}"
        res = self.collection.get(ids=[uid])
        if res and res["documents"]:
            return StoreData(namespace, key, {"data": res["documents"][0]})
        return None

    def search(self, namespace: tuple, query: str, top_k: int = 4):
        query_emb = embed(query)
        results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        return [StoreData(namespace, f"result{i}", {"data": doc}) for i, doc in enumerate(docs)]

# Initialize Chroma client and collection:
settings = Settings(persist_directory="chromadb_store")
chroma_client = chromadb.Client(settings)
collection = chroma_client.get_or_create_collection(name="global_memory")
chroma_store = ChromaStore(collection)
persistent_store = chroma_store  # Use in our LangGraph workflows
4.2 Integrating ChromaStore in a Workflow Task
Use the store to enrich LLM prompts with long-term memory:

@task
def find_relevant_facts(query: str, store: ChromaStore, user_id: str) -> str:
    namespace = ("facts", user_id)
    results = store.search(namespace, query=query)
    if not results:
        return ""
    info = "\n".join(res.value["data"] for res in results)
    return f"Relevant info:\n{info}"
Call this task inside your workflows to dynamically fetch context that improves the LLM’s responses.

5. Storyboard Generation with Stable Diffusion API
Enhance narratives by generating visual storyboards. This section demonstrates generating prompts from narrative text and fetching images via a Stable Diffusion-compatible API.

5.1 Generating Storyboard Prompts and Retrieving Images
import requests
from typing import List

@task
def generate_storyboard_prompts(narrative: str) -> List[str]:
    # Assume each newline in narrative represents a scene description
    return [line.strip() for line in narrative.split('\n') if line.strip()]

@task
def fetch_images(prompts: List[str], api_url: str) -> List[str]:
    image_urls = []
    for prompt in prompts:
        res = requests.post(api_url, json={"prompt": prompt})
        image_urls.append(res.json().get("image_url"))
    return image_urls

@entrypoint
def storyboard_agent(narrative: str, api_url: str) -> List[str]:
    prompts = generate_storyboard_prompts(narrative).result()
    images = fetch_images(prompts, api_url).result()
    return entrypoint.final(value=images, save={})
5.2 Displaying Storyboard Images in Chainlit
@cl.on_message
async def handle_storyboard(message: cl.Message):
    config = load_config("config.yaml")
    out = director_agent(message.content, api_url=config.storyboard_api_url).result()
    await cl.Message(content=out["gm_response"], author="Game Master").send()
    for img_url in out.get("images", []):
        await cl.send_image(img_url, caption="Storyboard Scene")
6. Testing the Integration with Pytest
A robust test suite is vital. Below are examples to test your LangGraph workflows, Chainlit event handlers, and ChromaDB integration.

6.1 Unit Testing LangGraph Tasks
def test_call_llm(monkeypatch):
    monkeypatch.setattr("your_module.call_llm", lambda user_message, context: "Test LLM output")
    result = call_llm("Hello", "Context").result()
    assert "Test LLM" in result

def test_chat_workflow(monkeypatch):
    config = {"configurable": {"thread_id": "test123", "user_id": "u1"}}
    res1 = chat_workflow.invoke("remember: the sky is blue", config=config)
    assert "blue" in res1.lower()
    res2 = chat_workflow.invoke("what color is the sky?", config=config)
    assert "blue" in res2.lower()
6.2 Testing Chainlit Handlers (Logic Layer)
import asyncio
from chainlit.models import Message as ClMessage

async def test_handle_message(monkeypatch):
    cl.user_session.set("thread_id", "test_thread")
    cl.user_session.set("message_count", 0)
    monkeypatch.setattr("your_module.chat_workflow.invoke", lambda msg, config: "Test response")
    fake_message = ClMessage(content="Hello")
    await handle_message(fake_message)
6.3 Testing ChromaStore Integration
def test_chroma_store(tmp_path, monkeypatch):
    from chromadb.config import Settings
    settings = Settings(persist_directory=str(tmp_path))
    client = chromadb.Client(settings)
    collection = client.create_collection("testcol")
    test_store = ChromaStore(collection)
    test_store.put(("facts", "u1"), "fact1", {"data": "Test fact about AI"})
    results = test_store.search(("facts", "u1"), query="AI")
    assert any("ai" in r.value["data"].lower() for r in results)
6.4 Asynchronous Streaming Tests
import pytest

@pytest.mark.asyncio
async def test_streaming_response(monkeypatch):
    async def fake_stream(message, config, stream_mode):
        for chunk in ["Chunk1 ", "Chunk2 ", "Final"]:
            yield chunk, {"langgraph_node": "narration"}
    monkeypatch.setattr(chat_workflow, "stream", fake_stream)
    collected = ""
    async for chunk, meta in chat_workflow.stream("Hello", config={"configurable": {"thread_id": "stream_test", "user_id": "u1"}}, stream_mode=["tokens"]):
        collected += chunk
    assert "Final" in collected
7. Advanced Integration Use Cases and Best Practices
7.1 Integrating Multi-Agent Hierarchies
Hierarchical Agents: Use a Director agent to delegate to subordinate agents (GM, NPC, PC). Each agent is an independent entrypoint and can be chained.
Decoupled Communication: Have agents communicate via shared state (using Pydantic models) and persistent stores. This ensures that if one agent (e.g., NPC) requires background data (e.g., past interactions), it can retrieve that from the store.
Tool Invocation: Allow agents to call external tools (e.g., a dice roll or a memory search) by registering them with LangGraph. The ReAct pattern can be implemented: the agent outputs an action that triggers a tool task, and then the tool’s output is fed back.
7.2 Migrating to the Functional API
From Graph Nodes to Functions: Replace older static graph definitions by refactoring each node into a function decorated with @task or @entrypoint.
State as Return Values: Instead of relying on mutable global state, pass state as function parameters and return updated state using entrypoint.final.
Unified Debugging: With plain Python functions, use standard debugging techniques (logging, unit tests) to trace data flow.
7.3 Leveraging Chainlit’s UI Features
Session Persistence: Use Chainlit’s on_chat_start and on_chat_resume hooks to initialize and restore sessions. Always pass the same thread ID to ensure LangGraph state continuity.
Media Integration: Combine text, images, audio, and video to create immersive interfaces. Use Chainlit’s built-in message elements and actions for interactivity.
Interactive Options: Present multiple-choice options or commands to the user. Define callbacks to handle these actions.
7.4 Ensuring Robust Testing with Pytest
Unit and Integration Tests: Write tests for individual tasks and entire workflows. Use monkeypatching to simulate LLM responses and external API calls.
Asynchronous Testing: Utilize pytest.mark.asyncio to test streaming and asynchronous functions.
Regression Checks: Validate that changes in Pydantic models or workflow logic do not break existing functionality. Use sample outputs to enforce contracts.
Summary
This document serves as an extensive toolbox for integrating and extending interactive AI applications using:

LangGraph Functional API: Build modular workflows with persistence, easy migration, and asynchronous streaming.
Hierarchical Multi-Agent Architectures: Design systems where a Director agent orchestrates GM, NPC, and PC agents using decoupled communication.
Chainlit: Create an engaging UI with rich media, session management, message streaming, and interactive actions.
Pydantic: Enforce structured configuration and state models, providing type safety and robust data validation.
ChromaDB: Implement long-term vector memory via a custom store (e.g., ChromaStore) that integrates with your workflows.
Pytest: Ensure stability with comprehensive tests covering unit, integration, and asynchronous behaviors.
Each code snippet and example is designed to be adapted and extended as needed. This toolbox provides everything an AI coding agent requires to build a scalable, interactive system—whether for AI storytelling, interactive chat applications, or other conversational interfaces—without relying on external references later.

Use this document as your definitive reference, and feel free to extend or modify components as your application evolves.
