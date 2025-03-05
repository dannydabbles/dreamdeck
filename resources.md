# LangGraph Functional API and Chainlit: A Practical Guide

This guide covers the practical implementation of **LangGraph’s Functional API** in combination with **Chainlit** to build advanced LLM applications. We’ll explore how to design multi-agent workflows, integrate tool usage (including human-in-the-loop interactions), manage memory and state, persist conversation history in Chainlit, leverage Pydantic for structured data, write tests with pytest, and implement asynchronous workflows with streaming. Each section includes conceptual overviews **with code snippets** for clarity.

## 1. LangGraph Functional API Overview (Core Concepts)

LangGraph’s Functional API provides a flexible way to define AI workflows using Python functions and decorators. It introduces two key primitives: **`@entrypoint`** and **`@task`** ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=The%20Functional%20API%20consists%20of,having%20to%20restructure%20your%20code)) ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=Building%20Blocks)).

- **Task**: A discrete unit of work (e.g. an LLM call or a tool execution) defined as a function with `@task`. Invoking a task returns a future-like object that you can `.result()` to get the output (or await it asynchronously) ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=,the%20result%20or%20resolved%20synchronously)). Tasks can be composed and even run in parallel.
- **Entrypoint**: The top-level workflow function decorated with `@entrypoint`. It contains the main logic and can orchestrate tasks, handle control flow (loops, conditionals), manage long-running operations, and support interrupts (for human input). The entrypoint serves as the **starting point** of execution and manages persistence (state checkpointing) and possible resume of workflows ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=,the%20result%20or%20resolved%20synchronously)).

**Execution Flow**: When you call an entrypoint (via `.invoke()` or `.stream()`), it runs the workflow function. Inside, calling a task (e.g. `result = some_task(args)`) immediately returns a future. Calling `.result()` on the future will execute the task (and block until completion, unless done asynchronously). This allows writing sequential code that actually runs tasks asynchronously under the hood. You can use normal Python control flow in the entrypoint to decide which tasks to run or whether to loop, making the workflow logic very readable. The Functional API “under the hood” still builds on LangGraph’s graph runtime, so it benefits from features like persistence, memory, interrupts, and streaming without requiring explicit graph definitions ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=restructure%20your%20code)).

**Basic Example**: Here’s a simple entrypoint with two tasks to illustrate the structure:

```python
from langgraph.func import entrypoint, task

@task
def add(x: int, y: int) -> int:
    return x + y

@task
def double(z: int) -> int:
    return 2 * z

@entrypoint()
def workflow(values: dict) -> int:
    # values is a dict with keys "a" and "b"
    total = add(values["a"], values["b"])         # returns a future
    summed = total.result()                       # get actual result
    doubled = double(summed).result()             # call second task
    return doubled  # final result of workflow
```

This workflow adds two numbers and then doubles the sum. The key is that each `@task` can run independently. We could even parallelize tasks if needed (see the Async section). In a real scenario, tasks might call LLM APIs or other services.

**LangGraph Entrypoint Behavior**: An entrypoint can take arguments normally, but only the first positional argument is considered the workflow input (for multiple inputs, pass a dictionary or use keyword args) ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=When%20defining%20an%20,you%20can%20use%20a%20dictionary)). It can also accept special parameters like `previous` or `store` (discussed later for memory). When the entrypoint completes, you typically return the final result. If using persistence features, you might return using `entrypoint.final(...)` which lets you separate the value to output vs the value to save as state.

**Human-in-the-Loop and Interrupts**: The Functional API supports pausing execution to get human input via the `interrupt()` function and resuming with a `Command`. We’ll cover this in detail in the Tool Calling section. In essence, calling `interrupt(prompt)` inside a task will **halt the workflow** and yield control (so you can ask the user `prompt` via the UI). When the user provides input, you resume the workflow by calling the entrypoint again with a `Command(resume=...)` containing the response ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=)) ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=,n)). This mechanism, combined with automatic state persistence, makes implementing human-in-the-loop flows straightforward.

**Summary**: The Functional API lets you write LangChain/LangGraph workflows in a natural style – using Python functions for tasks and workflows – while still getting LangGraph’s powerful features like parallel task execution, state checkpointing, and human intervention. We’ll leverage these features in the following sections.

## 2. Multi-Agent Workflows with LangGraph Functional API

One powerful use of LangGraph is to coordinate **multiple agents** (LLM-powered actors) working together. In a multi-agent workflow, each agent can have its own role or expertise, and they can exchange information or delegate tasks to each other. LangGraph makes this possible by treating each agent as a task or node and managing the control flow between them ([How to build a multi-agent network (functional API)](https://langchain-ai.github.io/langgraphjs/how-tos/multi-agent-network-functional/#:~:text=In%20this%20how,defined%20in%20the%20main%20entrypoint)).

**Design Considerations**: When building a multi-agent system, consider:
- **Agents**: What independent agents do you have? (Each could be a separate prompt + LLM, possibly with specialized tools or knowledge.)
- **Agent communication pattern**: How do agents hand off or collaborate? (One agent might call another for help on certain queries, or all agents might share a common scratchpad of conversation.)

LangGraph’s graph metaphor suits this: each agent is a node, and edges define communication pathways ([LangGraph: Multi-Agent Workflows](https://blog.langchain.dev/langgraph-multi-agent-workflows/#:~:text=2,connected)) ([LangGraph: Multi-Agent Workflows](https://blog.langchain.dev/langgraph-multi-agent-workflows/#:~:text=This%20thinking%20lends%20itself%20incredibly,adding%20to%20the%20graph%27s%20state)). Using the Functional API, we typically create each agent as a task (or a set of tasks), and orchestrate their interaction in the entrypoint logic.

**Example – Two Collaborating Agents**: Imagine a **travel planning** assistant composed of two agents: one for travel destinations and one for hotel recommendations. They can call each other as needed. We define each agent as a LangChain ReAct agent with its own tools, and use a main workflow to route between them:

```python
from langgraph.func import entrypoint, task
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI

# Define the tools (for simplicity, dummy implementations)
def get_travel_recommendations() -> str:
    return "Try visiting Paris or Tokyo."
def get_hotel_recommendations(location: str) -> str:
    return f"Hotels in {location}: Hotel A, Hotel B."

# Wrap tools in LangChain’s tool format if needed, or define as tasks:
# (Here we assume initialize_agent will handle the tool calling via LangChain)

llm = OpenAI(...)  # some LLM initialization
travel_agent = initialize_agent(
    tools=[{"name": "GetTravelRecommendations", "func": get_travel_recommendations, "description": "Recommend travel destinations"}],
    llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
hotel_agent = initialize_agent(
    tools=[{"name": "GetHotelRecommendations", "func": get_hotel_recommendations, "description": "Recommend hotels in a location"}],
    llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Define tasks to invoke each agent
@task
def ask_travel_agent(user_message: str):
    """Send user query to travel agent and get response (which may include a tool use or final answer)."""
    return travel_agent.run(user_message)

@task
def ask_hotel_agent(user_message: str):
    """Send user query to hotel agent and get response."""
    return hotel_agent.run(user_message)

# Entrypoint to coordinate agents
@entrypoint()
def multi_agent_workflow(user_query: str) -> str:
    # Start with the travel agent by default
    conversation = []
    current_query = user_query
    next_agent = ask_travel_agent

    # Loop allowing agents to hand off between each other
    while True:
        response = next_agent(current_query).result()  # get agent response
        conversation.append(response)
        # Simple logic: if travel agent decided to hand off to hotel agent (say by a special signal in response):
        if "transfer to hotel agent" in response.lower():
            # Switch to hotel agent for next turn
            next_agent = ask_hotel_agent
            # Perhaps extract a specific query for the hotel agent, here assumed as part of response:
            current_query = "best hotels in Paris"  # (In practice, parse from response)
            continue
        break  # if no handoff signal, break loop

    final_answer = response  # last agent's response
    return final_answer
```

In the above pseudo-code, `ask_travel_agent` and `ask_hotel_agent` are tasks wrapping calls to each agent. The `multi_agent_workflow` entrypoint sends the user query to the travel agent first. If the travel agent’s answer indicates handing off (we used a placeholder check for a phrase like "transfer to hotel agent"), the workflow then directs the next question to the hotel agent. The loop continues until an agent produces a final answer without handing off.

**Key Patterns**:
- Each agent is encapsulated as a callable (could be a LangChain agent or a LangGraph sub-workflow).
- The entrypoint manages which agent is “active” and when to switch. In this case, we used a simple string signal; in more complex cases, the agents could explicitly output an action like `{"action": "handoff_to", "agent": "hotel", ...}` that we parse.
- We maintain a `conversation` list if needed to log the interaction. LangGraph could also maintain state (more on that in Memory Management section).

LangGraph’s official docs demonstrate a similar idea where a *tool* is used to signal agent handoff, and the entrypoint inspects the conversation to choose the next agent ([How to build a multi-agent network (functional API)](https://langchain-ai.github.io/langgraphjs/how-tos/multi-agent-network-functional/#:~:text=In%20this%20how,defined%20in%20the%20main%20entrypoint)) ([How to build a multi-agent network (functional API)](https://langchain-ai.github.io/langgraphjs/how-tos/multi-agent-network-functional/#:~:text=let%20callActiveAgent%20%3D%20callTravelAdvisor%3B%20let,return%20messages%3B)). The benefit of using LangGraph tasks for each agent is that the handoff and looping logic can be written in straightforward Python, while LangGraph handles the execution and state under the hood.

**Many-to-Many Agent Networks**: The above is a one-to-one handoff. LangGraph can handle more complex networks (e.g. multiple agents all communicating via a shared message board, or hierarchical setups). For instance, a **fully connected network** where any agent can call any other would involve more dynamic routing. Typically you might give each agent a tool (or an action) to call each other agent, and in the workflow loop route accordingly. The Functional API’s flexibility allows implementing custom coordination schemes (like round-robin questioning, or a central “manager” agent delegating tasks to specialist sub-agents, etc.).

**Benefits**: Multi-agent workflows can break complex tasks into specialized subtasks, improving performance and reliability ([LangGraph: Multi-Agent Workflows](https://blog.langchain.dev/langgraph-multi-agent-workflows/#:~:text=,without%20breaking%20the%20larger%20application)) ([LangGraph: Multi-Agent Workflows](https://blog.langchain.dev/langgraph-multi-agent-workflows/#:~:text=Multi,specialized%20agents%20and%20LLM%20programs)). LangGraph provides a structured way to design these interactions (each agent as a node and transitions as edges in concept), while the Functional API lets you implement it in code without explicitly drawing a graph.

## 3. Tool Calling in the Functional API (ReAct Agents & Human-in-the-Loop)

Integrating external tools (e.g. web search, calculators, databases) is a common requirement in LLM applications. LangGraph’s Functional API supports tool usage, and you can incorporate **human as a tool** for human-in-the-loop interactions.

**Defining Tools**: In LangChain/LangGraph, a “tool” is typically a function that the agent can call. In the Functional API (Python), you can use the `@tool` decorator from LangChain to wrap a function as a tool, or simply define a normal function and ensure the agent knows about it. Each tool should have a **name**, **description**, and an **input schema** (which can be enforced via Python type hints, Pydantic, or even simple dict schemas). For example:

```python
from langchain.agents import tool

@tool
def search_web(query: str) -> str:
    """Search the web for the query and return the top result."""
    # ... implementation ...
    return result_text
```

The above would allow an agent to call `search_web` if it decides to. When using LangGraph, how do we plug tools in? One approach is to use LangChain’s agent tooling in tasks. Another approach is to manually manage tool calls.

**ReAct Loop via Tasks**: A common pattern (as seen in LangChain’s ReAct agent) is:
1. Call LLM to get an action (tool invocation or final answer).
2. If the LLM requested a tool, execute the tool and add the result to the context.
3. Repeat until LLM provides a final answer.

We can implement this loop with Functional API tasks ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=%40task%20def%20call_model%28messages%29%3A%20,invoke%28messages%29%20return%20response)) ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=,result%28%29%20for%20fut%20in%20tool_result_futures)):
- A task to call the model and get an output (which may include tool usage instructions).
- A task to call a tool and return its result as a message.
- The entrypoint to orchestrate the loop.

For example:

```python
from langgraph.func import entrypoint, task
from langchain.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

# Define a tool using LangChain's @tool
@tool
def get_weather(location: str) -> str:
    """Get weather for a given location."""
    # dummy implementation:
    if location.lower() in ["sf", "san francisco"]:
        return "It's sunny."
    elif location.lower() == "boston":
        return "It's rainy."
    else:
        return f"No weather info for {location}."

# Define an "LLM + tools" instance (LangGraph can use LangChain models under the hood)
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [get_weather]  # list of tool functions
tools_by_name = {t.name: t for t in tools}

@task
def call_model(messages):
    """Call the LLM with accumulated messages (chat history) and bound tools."""
    # Bind tools so the LLM knows it can call them (for openAI functions or LangChain agent)
    response = model.bind_tools(tools).invoke(messages)
    return response  # response might include llm_output plus tool_calls list

@task
def call_tool(tool_call: dict):
    """Execute a tool based on the LLM's tool call instruction."""
    tool_name = tool_call["name"]
    tool_input = tool_call.get("args") or tool_call.get("input")
    result = tools_by_name[tool_name].invoke(tool_input)
    # Wrap the tool’s output as a message to feed back to LLM (ToolMessage in LangChain)
    return ToolMessage(content=result, tool_call_id=tool_call["id"])
```

In this snippet, `model.bind_tools(tools).invoke(messages)` calls the LLM and allows it to choose a tool defined in `tools` ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=%40task%20def%20call_model%28messages%29%3A%20,invoke%28messages%29%20return%20response)). The returned `response` would likely include a `response.tool_calls` attribute (a list of tool call instructions the LLM decided on). We then loop over those tool calls, execute each with `call_tool`, and append the results to the message list.

**Tool Loop in Entrypoint**: The entrypoint ties it together:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def agent(messages: list, *, previous=None):
    # If there's conversation history from a previous run, include it
    if previous is not None:
        messages = add_messages(previous, messages)
    # Call LLM
    llm_response = call_model(messages).result()
    # Loop while LLM has tool actions
    while llm_response.tool_calls:
        tool_futures = [call_tool(tc) for tc in llm_response.tool_calls]
        tool_results = [f.result() for f in tool_futures]  # execute all requested tools in parallel
        # Add the tool results (as messages) to the conversation and call LLM again
        messages = add_messages(messages, [llm_response, *tool_results])
        llm_response = call_model(messages).result()
    # No more tool calls; final answer ready
    messages = add_messages(messages, llm_response)
    # Return final answer, and save full conversation in state (for memory)
    return entrypoint.final(value=llm_response, save=messages)
```

This structure implements a ReAct agent using tasks ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=while%20True%3A%20if%20not%20llm_response,break)) ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=,result)):
1. Combine new user message with previous conversation (`previous` comes from memory, if any).
2. Call the LLM (`call_model`) to get a response.
3. If the response includes tools to call, execute each tool via `call_tool` tasks (note: we collect futures and then `.result()` them, which allows parallel tool execution if multiple are issued) ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=,result%28%29%20for%20fut%20in%20tool_result_futures)) ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=,result)).
4. Append the LLM response and tool results to the messages, and loop back to call the LLM again with the updated context.
5. Once the LLM produces no tool calls, that response is the final answer. We add it to messages and return, using `entrypoint.final` to save the entire conversation (for memory continuity).

**Human-in-the-Loop as a Tool**: Sometimes the “tool” an agent should use is asking the **user** for input (e.g. clarifying a query, or, as in the Reddit example, asking for credentials mid-workflow). We can implement this by creating a tool that triggers an interrupt. LangGraph’s `interrupt()` function will pause the workflow and yield control back to us with a prompt for the user ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=Human,and%20await%20input%20before%20proceeding)) ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=Waiting%20for%20human%20input%20is,and%20await%20input%20before%20proceeding)).

For instance, define a tool that when called by the agent, stops and asks the user:

```python
from langgraph.types import interrupt
from langchain.agents import tool

@tool
def human_assistance(query: str) -> str:
    """Tool that asks the user for assistance with the given query."""
    # When agent calls this tool, pause execution and ask user
    user_response = interrupt({"query": query})
    # The `interrupt` returns a dict (or value) after resume. Suppose user provides {"data": "..."}.
    return user_response["data"]
```

Adding this `human_assistance` tool to the agent’s tool list means the LLM can decide to “invoke” a human helper ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=from%20langgraph)). When `interrupt()` is called inside the task, LangGraph will produce an `__interrupt__` event containing the prompt or data needed ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=)). At this point, the workflow **pauses** and awaits a resume signal.

**Using Interrupt with Chainlit**: Chainlit doesn’t automatically know what to do with LangGraph’s interrupt events, so you’ll integrate it manually:
- Run the agent via `entrypoint.stream()` in your Chainlit app. As you iterate over events, check for an event indicating an interrupt (it might appear as a dictionary with `{"__interrupt__": ...}`).
- When you detect an interrupt, you can use Chainlit’s **AskUserMessage** to prompt the user for input in the UI ([Issues when prompting for credentials using Chainlit UI + Langgraph : r/LangChain](https://www.reddit.com/r/LangChain/comments/1h3u7ei/issues_when_prompting_for_credentials_using/#:~:text=%E2%80%A2)). For example, in your Chainlit `on_message` handler:

  ```python
  import chainlit as cl

  @cl.on_message
  async def handle_message(message: str):
      for event in agent.stream([{"role": "user", "content": message}], config={"configurable": {"thread_id": "xyz"}}):
          if "__interrupt__" in event:
              prompt = event["__interrupt__"][0].value  # the message asking for input
              # Use Chainlit AskUserMessage to prompt user on UI and wait for reply
              user_reply = await cl.AskUserMessage(content=prompt).send()
              # Resume the agent by sending the Command with user reply
              resume_event = agent.stream(cl.Command(resume=user_reply["content"]), config={"configurable": {"thread_id": "xyz"}})
              # continue iterating over resume_event and so on
          else:
              # handle normal events (LLM responses, tool outputs, etc.)
              await cl.Message(content=str(event)).send()
  ```
  *(Pseudocode for demonstration)*

  The idea is: when LangGraph interrupts, we present the user with a question. The user's answer is then passed back into the workflow via `Command(resume=...)` to continue ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=provides%20instructions%20to%20resume%20the,task)). Chainlit’s `AskUserMessage` is designed exactly for these scenarios – it displays a message and waits for the user’s response, allowing a seamless human-in-loop interaction in the UI ([Issues when prompting for credentials using Chainlit UI + Langgraph : r/LangChain](https://www.reddit.com/r/LangChain/comments/1h3u7ei/issues_when_prompting_for_credentials_using/#:~:text=%E2%80%A2)).

**Best Practices for Tool Integration**:
- Define clear tool schemas. Use Pydantic models or dataclasses for complex tool inputs to ensure the agent gets structured data (LangChain’s function calling is compatible with Pydantic schemas; more in Pydantic section).
- Avoid too many tools per agent; it’s often better to split into multiple agents if you have a very large toolset ([LangGraph: Multi-Agent Workflows](https://blog.langchain.dev/langgraph-multi-agent-workflows/#:~:text=,without%20breaking%20the%20larger%20application)).
- For human tools, ensure the agent’s prompt or logic knows when to invoke them (e.g., the agent could have an instruction like “If you need help from a human, call the `human_assistance` tool”). Also handle the user response carefully – possibly re-validating it or confirming.
- **Human approval flows**: Another human-in-loop pattern is reviewing a tool’s output. LangGraph provides mechanisms to *review or edit* tool outputs via interrupts as well (using `Command` to approve or modify results) ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=match%20at%20L619%20%2A%20Human,determinism%20section%20for%20more%20details)) ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=match%20at%20L652%20To%20utilize,deterministic)), which can be useful for safety-critical applications.

By combining LangGraph Functional API’s tool calling with Chainlit’s UI for human input, you can build rich interactive agents that not only use automated tools but also seamlessly defer to humans when necessary.

## 4. Memory State Management in LangGraph (Multi-Turn Conversations)

Maintaining state across turns (or steps) is crucial for conversational agents and multi-step workflows. LangGraph’s Functional API provides built-in support for **short-term memory** (conversation or session state) and **long-term memory** (persistent data across sessions) ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=Short)) ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=You%20can%20implement%20long,interactions%20with%20the%20same%20user)).

### 4.1 Short-Term Memory (Conversation Context)

Short-term memory refers to the running context of the current conversation or workflow run – for example, the dialogue history in a chat with a user. In LangGraph, enabling persistence on an entrypoint allows the framework to store the state after each run and feed it into the next run of that workflow for the same “session.” The Functional API offers two mechanisms to help with this:
- The `previous` parameter in an entrypoint function signature.
- The `entrypoint.final()` return helper.

When you decorate an entrypoint with a `checkpointer` (like `MemorySaver` for in-memory persistence), you can include a parameter named `previous` in the function. LangGraph will automatically fill this with the saved state from the last execution (if any) for the same conversation thread ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=In%20the%20Functional%20API%2C%20you,term%20memory%20using)) ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=%40entrypoint%28checkpointer%3Dcheckpointer%29%20def%20conversational_agent%28user_message%2C%20,messages%20%3D%20previous%20or)). The saved state itself is determined by what you return via `entrypoint.final`.

**Example**: Simplified conversational agent entrypoint using these features:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def chat_agent(user_input: str, *, previous=None):
    # previous contains the conversation list from last turn, if exists
    conversation = previous or []  # start with old messages or fresh list
    conversation.append({"role": "user", "content": user_input})
    # Call LLM with the full conversation (this could be via a task, omitted for brevity)
    assistant_reply = call_llm(conversation)  # suppose this returns assistant message text
    conversation.append({"role": "assistant", "content": assistant_reply})
    # Return the assistant's reply as output, but save the entire conversation
    return entrypoint.final(value=assistant_reply, save=conversation)
```

Here:
- We set `checkpointer=MemorySaver()`, which tells LangGraph to keep a checkpoint of the state in memory for each unique session (the session is identified by a `thread_id` in the config when calling, e.g. we might call this entrypoint with `config={"configurable": {"thread_id": "session1"}}`).
- `previous` is used to retrieve the last saved conversation for the session ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=%40entrypoint%28checkpointer%3Dcheckpointer%29%20def%20conversational_agent%28user_message%2C%20,messages%20%3D%20previous%20or)).
- We append the new user message to the conversation, then generate the assistant’s reply (through some LLM call).
- We return `entrypoint.final(value=assistant_reply, save=conversation)`. The `entrypoint.final` tells LangGraph: the immediate result of this workflow is `assistant_reply` (just the assistant's answer), but the state to persist (`previous` for next time) is the full `conversation` list ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=,extend%28new_messages)). This way, next time the user sends a message in the same session, `previous` will contain all past messages, allowing the agent to maintain context.

Under the hood, LangGraph will store the `conversation` in a checkpoint (MemorySaver keeps it in memory; other checkpointers can serialize to disk or database). On the next invocation for the same `thread_id`, LangGraph injects that saved conversation as `previous`.

**Multi-Turn Interaction**: With this setup, you can handle a dialogue iteratively. For example:

```python
# First user message
agent.invoke("Hello", config={"configurable": {"thread_id": "session1"}})
# returns "assistant's first reply"

# Second user message
agent.invoke("What's the weather?", config={"configurable": {"thread_id": "session1"}})
# Now inside chat_agent, previous will contain [{"role": "user": ...}, {"role": "assistant": ...}] from first turn.
# The agent can use that context before answering the second question.
```

LangGraph ensures that tasks already executed in a prior turn are not needlessly repeated and that the sequence of messages is preserved ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=Tip)) ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=)). The built-in persistence abstracts away a lot of boilerplate; you don’t need to manually manage a conversation buffer – just use `previous` and `entrypoint.final(save=...)`.

### 4.2 Long-Term Memory (Cross-Session Persistence)

Long-term memory means storing information that persists beyond a single conversation thread. For instance, a personal assistant might remember a user’s preferences or profile information across all sessions. LangGraph addresses this with a concept of **stores**.

You can provide a `store` to an entrypoint (via the `store` argument in the decorator) which gives you access to a key–value storage (could be in-memory, database-backed, etc.) ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=You%20can%20implement%20long,interactions%20with%20the%20same%20user)) ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=You%20can%20implement%20long,interactions%20with%20the%20same%20user)). The `store` typically implements methods like `.get(key)` and `.set(key, value)` to read/write persistent data.

To use long-term memory:
```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()  # or a persistent implementation

@entrypoint(checkpointer=MemorySaver(), store=store)
def agent_with_long_term(data: dict, *, previous=None, store=None):
    # 'store' is automatically provided if named as parameter
    user_id = data.get("user_id")
    # Retrieve user profile or preferences
    profile = store.get(f"profile_{user_id}") or {}
    # ... use profile info in prompt or logic ...
    # maybe update something
    store.set(f"profile_{user_id}", profile)
    # normal short-term handling...
    return entrypoint.final(value=..., save=...)
```

Here, `store: BaseStore` parameter in the function is injected by LangGraph ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=match%20at%20L183%20from%20langgraph,memory%20import%20InMemoryStore)) ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=You%20can%20implement%20long,interactions%20with%20the%20same%20user)). We can call `store.get` or `store.set` to retrieve or update persistent data. Unlike `previous` which is tied to a specific conversation thread, `store` is shared and survives independently (in this case, since we used `InMemoryStore`, it lasts as long as the app runs; for real persistence you might use a file-based or database store).

Use cases for long-term memory:
- Storing user-specific data (preferences, past interactions summary, domain knowledge) that should be available even if the conversation restarts.
- Caching expensive results (like embeddings, or tool outputs) keyed by some identifier, to reuse later.
- Implementing a simple vector store or knowledge base by writing to the store across sessions.

LangGraph’s design cleanly separates short-term vs long-term: short-term (conversation state) is handled via checkpointing/previous, and long-term via store. This separation prevents confusion between ephemeral dialogue context and truly persistent knowledge.

**Memory and Determinism**: It’s worth noting that LangGraph aims to make workflows reproducible. If randomness (like an LLM call) is involved, the checkpointing ensures that once a task’s result is obtained, it’s reused on resume rather than re-run, to avoid divergence ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=match%20at%20L652%20To%20utilize,deterministic)). This is critical when using interrupts and memory – you want the conversation so far to remain fixed when resuming. LangGraph’s memory system takes care of that by persisting prior task results after an interrupt ([How to wait for user input (Functional API)](https://langchain-ai.github.io/langgraph/how-tos/wait-user-input-functional/#:~:text=Tip)).

**Alternate Memory Options**: `MemorySaver` is an in-memory checkpoint; for production you might use a more durable checkpointer (there might be DiskSaver or database-backed savers in LangGraph). Ensure the choice fits your deployment – for example, if you run multiple server replicas, a memory checkpointer won’t share state between them, so a centralized store or DB might be needed for consistency.

In summary, LangGraph provides out-of-the-box solutions for maintaining conversation state and storing persistent data:
- Use `previous` and `entrypoint.final(..., save=state)` for keeping track of multi-turn interactions **within a session** ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=%40entrypoint%28checkpointer%3Dcheckpointer%29%20def%20conversational_agent%28user_message%2C%20,messages%20%3D%20previous%20or)) ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=,extend%28new_messages)).
- Use `store` for **cross-session memory** that all sessions (or future sessions) can access ([Introducing the LangGraph Functional API](https://blog.langchain.dev/introducing-the-langgraph-functional-api/#:~:text=You%20can%20implement%20long,interactions%20with%20the%20same%20user)).
- These mechanisms free you from manually plumbing state through every function or using global variables, and they play nicely with multiple users (each conversation thread is uniquely identified, avoiding cross-talk).

## 5. Memory Persistence in Chainlit (Conversation History Across Sessions)

Chainlit provides features to **persist chat history** and user session data, which complement LangGraph’s in-app memory. By default, Chainlit does *not* persist conversations – if you refresh or restart, the chat log is gone. However, with a bit of configuration, Chainlit can store chats (e.g., in a database) and allow users to resume them later ([Overview - Chainlit](https://docs.chainlit.io/data-persistence/overview#:~:text=By%20default%2C%20your%20Chainlit%20app,of%20your%20project%20or%20organization)) ([Overview - Chainlit](https://docs.chainlit.io/data-persistence/overview#:~:text=Open%20Source%20Data%20Layer%20Use,28)).

### 5.1 Enabling Chat Persistence in Chainlit

To turn on persistence, you need to set up a **data layer** in Chainlit:
- Use Chainlit’s **official data layer** or a community one. The official data layer uses a PostgreSQL (or SQLite) database via SQLAlchemy to store conversations, messages, and feedback.
- Set the environment variable `DATABASE_URL` in your Chainlit app’s environment (for example, in a `.env` file) to point to your database ([Official Data Layer - Chainlit](https://docs.chainlit.io/data-layers/official#:~:text=steps%2C%20feedback%2C%20etc)). For a quick local setup, you might use SQLite: `DATABASE_URL=sqlite:///chainlit.db`.
- Ensure you have authentication enabled (Chainlit supports simple auth or OAuth) if you want multiple users to have separate histories. Chainlit requires both persistence and user authentication to properly attribute and retrieve chat histories ([Chat History - Chainlit](https://docs.chainlit.io/data-persistence/history#:~:text=Chat%20history%20allow%20users%20to,and%20resume%20their%20past%20conversations)) ([Chat History - Chainlit](https://docs.chainlit.io/data-persistence/history#:~:text=If%20data%20persistence%20is%20enabled,to%20see%20the%20chat%20history)).

Once configured, Chainlit will save every conversation (thread) to the database. It also offers a UI for users to browse their past chats if logged in.

### 5.2 Resuming Conversations with on_chat_resume

To let a user pick up an old conversation (for example, continue where they left off yesterday), Chainlit provides a lifecycle hook `@cl.on_chat_resume`. This is called when a user opens a saved thread. Chainlit will automatically retrieve the stored messages and even restore any user session data that was saved ([on_chat_resume - Chainlit](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-chat-resume#:~:text=Decorator%20to%20enable%20users%20to,46%20to%20be%20enabled)) ([on_chat_resume - Chainlit](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-chat-resume#:~:text=,Restore%20the%20user%20session)). However, **for the LangGraph agent to actually continue the logic**, you might need to reconstruct its state.

Chainlit will by default **replay the messages to the UI** (so the user sees the chat history) and restore `cl.user_session` data. But if your agent relies on internal state (like LangGraph’s `previous` conversation or other memory), you need to handle that on resume.

**Steps to handle resume**:
1. **Identify the thread**: Chainlit passes a `thread: ThreadDict` object to the `on_chat_resume` handler, containing the conversation data (messages, etc.).
2. **Reinitialize or restore agent**: You may need to load the LangGraph memory for that thread. If you used a persistent checkpointer (like a database or MemorySaver that’s still in memory), LangGraph might still have the state keyed by `thread_id`. If not (e.g., your app restarted and MemorySaver lost data), you might reconstruct the `previous` messages from `thread`.
3. **Set session variables**: Chainlit’s `cl.user_session` can store custom info. On resume, it’s restored, but if your LangGraph agent object or some references weren’t kept, you might recreate them here.

A simple example using `on_chat_resume`:
```python
import chainlit as cl

@cl.on_chat_resume
async def on_chat_resume(thread: cl.ThreadDict):
    # This runs when a user resumes a saved chat.
    # thread contains keys like "messages": [message_dicts], etc.
    history = thread.get("messages", [])
    # If using LangGraph with thread IDs, ensure to use the same thread_id for continuity.
    # For example, store the thread_id in user_session on first run.
    saved_thread_id = cl.user_session.get("thread_id")
    if saved_thread_id:
        print(f"Resuming LangGraph session: {saved_thread_id}")
    else:
        # If not stored, you might derive it or set a new one
        cl.user_session.set("thread_id", thread["id"])
```

In practice, if you used `thread_id` in LangGraph config (like `"1"` or a random UUID) that might not directly correspond to Chainlit’s internal thread ID. One strategy is to tie them together:
- When starting a new chat, generate a unique ID (or use `thread["id"]` from Chainlit if accessible in `on_chat_start`) and use it as LangGraph’s `thread_id` for the checkpointer.
- Save this mapping somewhere (like `cl.user_session.set("thread_id", lg_id)`).
- On resume, retrieve it and pass it to LangGraph.

Chainlit’s resume decorator notes that if using a LangChain agent, you must re-instantiate it on resume ([on_chat_resume - Chainlit](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-chat-resume#:~:text=However%2C%20if%20you%20are%20using,in%20the%20user%20session%20yourself)). The same applies to a LangGraph agent: ensure your `agent` (entrypoint function object) is available and any needed initialization (API keys, etc.) is done. Typically, if your LangGraph setup is defined at module import, it will still be there.

**Integration with LangGraph Memory**: The ideal scenario is using a persistent LangGraph checkpointer (like one that writes to the same database Chainlit uses, or another DB). Then even if the app restarts, LangGraph can retrieve the conversation by thread_id. If that’s complex, a shortcut is to use Chainlit’s stored messages:
- The `thread["messages"]` list will contain the conversation (likely as a list of dict with roles and content).
- You can feed that into your LangGraph entrypoint on the next user query as the `previous` argument manually. For example, in `on_message`, if `cl.user_session.get("resumed")` is True, you might skip the normal memory retrieval and instead use the chainlit `history` to initialize LangGraph.

Chainlit basically handles the front-end of memory (display and user session), whereas LangGraph handles the back-end of memory (state for reasoning). You may combine them by:
  - Storing minimal state in Chainlit’s `user_session` (like an ID or some flags).
  - Letting Chainlit’s DB keep the message history for user visibility.
  - Relying on LangGraph’s `previous` via a persistent thread_id for actual context when calling the agent.

**Human Memory Example**: If a user resumes and asks a follow-up, you’d call:
```python
thread_id = cl.user_session.get("thread_id") or "default"
result = agent.invoke(user_input, config={"configurable": {"thread_id": thread_id}})
```
Because `thread_id` is the same as a previous conversation, LangGraph’s checkpointer will load `previous` messages. If not found (first run), `previous` will be None.

**Important**: If Chainlit’s chat history and LangGraph’s state diverge (e.g., you didn’t persist LangGraph state and app restarted), the agent might start fresh while the UI shows old messages. To avoid confusion, ensure one of:
- The agent can reconstruct context from the visible history (maybe by concatenating it into the prompt anew).
- Or use a persistent LangGraph memory tied to Chainlit’s chat storage.

Chainlit’s documentation and examples (see the “Resume Langchain Chat Example”) can guide how to load context on resume ([on_chat_resume - Chainlit](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-chat-resume#:~:text=However%2C%20if%20you%20are%20using,in%20the%20user%20session%20yourself)). The main point is that Chainlit gives you the hooks and storage to track conversations across sessions, and you as the developer need to link that with your agent’s memory mechanism.

### 5.3 Using Chainlit’s User Session for State

Apart from conversation content, Chainlit’s `cl.user_session` is a Python dictionary-like store that persists across the lifetime of a chat session (even across multiple messages). This is handy for lightweight state, counters, or caching inside your Chainlit app logic ([User Session - Chainlit](https://docs.chainlit.io/concepts/user-session#:~:text=import%20chainlit%20as%20cl)) ([User Session - Chainlit](https://docs.chainlit.io/concepts/user-session#:~:text=cl.user_session.set%28)). By default, Chainlit itself uses it to store things like the current chat thread. Only JSON-serializable data in `user_session` is saved across resumes ([on_chat_resume - Chainlit](https://docs.chainlit.io/api-reference/lifecycle-hooks/on-chat-resume#:~:text=,Restore%20the%20user%20session)).

For example, you might store a LangGraph “run id” or partial results in `cl.user_session` if needed. But avoid putting large or complex objects there.

**Summary**: To persist conversations in Chainlit:
- Enable a data layer with `DATABASE_URL` and possibly authentication ([Official Data Layer - Chainlit](https://docs.chainlit.io/data-layers/official#:~:text=steps%2C%20feedback%2C%20etc)).
- Use `@cl.on_chat_resume` to hook the resume event and restore any necessary agent state (like setting the proper LangGraph thread_id or reloading profile info).
- Ensure your agent calls use a consistent identifier for memory (so that the context continues).
- Test the resume flow: start a chat, get some conversation, stop and resume, ask another question – verify the agent still remembers context correctly.

By combining LangGraph’s memory (for the reasoning part) with Chainlit’s persistence (for the interface and session tracking), users can have continuous conversations that survive app restarts and can be recalled later.

## 6. Pydantic Integration with LangGraph (Type Validation & Structured Data)

Pydantic is a popular library for data validation and settings management, and it integrates nicely with LangChain/LangGraph for enforcing structured data schemas. There are a few areas where Pydantic models can enhance LangGraph workflows:

- **Tool Input/Output schemas**: Define tools to accept and return Pydantic `BaseModel` types for clarity and validation.
- **State validation**: If you’re using the Graph API (StateGraph) or even passing around complex state in Functional API, Pydantic can ensure at runtime that the state adheres to a schema ([How to use Pydantic model as graph state](https://langchain-ai.github.io/langgraph/how-tos/state-model/#:~:text=,can%20be%20any%20type)).
- **Model outputs**: Enforcing LLM outputs to match a schema (LangChain offers `PydanticOutputParser` to help LLMs format their output as a Pydantic model ([Pydantic parser - ️ LangChain](https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/pydantic/#:~:text=Pydantic%20parser%20,that%20conform%20to%20that%20schema)), which can then be parsed into actual model instances).

### 6.1 Using Pydantic for Tool Schemas

When defining a tool function, you can use Pydantic models to describe its input or output instead of raw dictionaries. For example:

```python
from pydantic import BaseModel

class WeatherRequest(BaseModel):
    location: str
    unit: str = "fahrenheit"

class WeatherInfo(BaseModel):
    location: str
    temperature: float
    condition: str

@tool
def get_weather(data: WeatherRequest) -> WeatherInfo:
    """Get weather info for a location."""
    # data is an instance of WeatherRequest, already validated
    loc = data.location
    # ... call some API ...
    return WeatherInfo(location=loc, temperature=72.0, condition="Sunny")
```

Using this approach, if the LLM tries to call `get_weather` with a wrong schema (say missing `location` or wrong type), LangChain/LangGraph can catch it. In LangChain’s function calling, if you provide a Pydantic model, it knows the JSON schema for it and the LLM’s output is validated against it. If the model returns a Pydantic object, Chainlit (or your code) can easily serialize it to show results.

### 6.2 Pydantic for State Validation (Graph API)

In LangGraph’s Graph API (not strictly needed for Functional API usage, but instructive), you can specify a `state_schema` when building a state graph. Pydantic models can serve as that schema. The LangGraph docs show how a `BaseModel` can be the state schema and thereby validate node inputs at runtime ([How to use Pydantic model as graph state](https://langchain-ai.github.io/langgraph/how-tos/state-model/#:~:text=In%20this%20how,run%20time%20validation%20on%20inputs)) ([How to use Pydantic model as graph state](https://langchain-ai.github.io/langgraph/how-tos/state-model/#:~:text=Known%20Limitations)). For example:

```python
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

class OverallState(BaseModel):
    a: str  # expecting a string field 'a'

def node(state: OverallState):
    # state is an OverallState instance
    return {"a": "processed"}  # returning an update

builder = StateGraph(state_schema=OverallState)
builder.add_node(node)
builder.add_edge(START, "node"); builder.add_edge("node", END)
graph = builder.compile()

# Test with valid input
print(graph.invoke({"a": "hello"}))  # {'a': 'processed'}

# Test with invalid input (a should be str, not int)
try:
    graph.invoke({"a": 123})
except Exception as e:
    print("Got validation error:", e)
```

Output:
```
Got validation error: 1 validation error for OverallState
a
  Input should be a valid string [type=string_type, input_value=123, input_type=int]
``` ([How to use Pydantic model as graph state](https://langchain-ai.github.io/langgraph/how-tos/state-model/#:~:text=except%20Exception%20as%20e%3A%20print%28,print%28e)) ([How to use Pydantic model as graph state](https://langchain-ai.github.io/langgraph/how-tos/state-model/#:~:text=An%20exception%20was%20raised%20because,9%2Fv%2Fstring_type))

The above shows Pydantic catching an invalid type for the state before the node runs. This kind of validation is useful to catch mistakes early (for instance, if an upstream LLM provided data not conforming to expected schema, you’d know).

While this example uses the explicit Graph API, the same concept can apply in Functional API if you design it similarly: e.g., define a Pydantic model for inputs and outputs of tasks. When a task is called, LangGraph will attempt to serialize inputs/outputs for checkpointing – if they are Pydantic, it should handle them as normal dataclasses.

**Pydantic in Functional Tasks**: You can annotate a task function’s parameters or return type with Pydantic models. LangGraph will treat those like any Python object. You won’t get automatic validation unless you actually create an instance of the model (or use Pydantic’s `validate` methods). One approach is to accept a dict but immediately parse it into a Pydantic model inside the task. For example:

```python
@task
def process_user_info(info: dict) -> dict:
    user = UserProfile(**info)  # Pydantic model parse & validation
    # now user is a validated model
    updated = user.copy(update={"last_seen": datetime.utcnow()})
    return updated.dict()
```

This way, if `info` is missing required fields, the Pydantic constructor will raise and LangGraph can handle the exception or fail early.

### 6.3 Structured Outputs from LLMs

Pydantic models shine when you want the LLM to return structured data. With function calling (OpenAI functions or LangChain’s agent tools) and output parsers, you can guide the LLM to fill a Pydantic schema. For instance, define a Pydantic for an answer format, and use LangChain’s `PydanticOutputParser` to format the prompt. This isn’t LangGraph-specific, but you can integrate it within a LangGraph task that calls the LLM. The result will be a Pydantic model (or easily parseable JSON) which you can then feed into subsequent tasks or tools reliably.

**Pydantic for Config**: LangGraph might also use Pydantic for configuration objects (e.g., LangSmith or other integration settings). If you encounter that, the principle remains – Pydantic ensures those configs are correct.

**Caveats**:
- Ensure Pydantic version compatibility. LangChain might use Pydantic v1 (via pydantic.v1 module if installed with v2) ([How to use LangChain with different Pydantic versions](https://python.langchain.com/v0.2/docs/how_to/pydantic_compatibility/#:~:text=How%20to%20use%20LangChain%20with,v1%20namespace%20of%20Pydantic%202)). If you encounter version issues, refer to LangChain docs on Pydantic integration.
- Pydantic validation is at runtime and adds some overhead. Use it where the safety or clarity benefits outweigh performance costs (usually fine unless you’re calling it thousands of times per second).
- The error messages from Pydantic might not automatically propagate in agent output. They will show up in exceptions/logs. In testing, you might assert that no validation errors occurred.

In summary, Pydantic helps add a **type safety net** to LangGraph:
- Use it to define structured **input/output schemas** for tasks and tools so that your LLM and tools communicate with well-defined data shapes.
- Use it to define the shape of shared **state** to catch any inconsistencies (especially in multi-agent or multi-step flows where many parts read/write the state) ([How to use Pydantic model as graph state](https://langchain-ai.github.io/langgraph/how-tos/state-model/#:~:text=1%20validation%20error%20for%20OverallState,9%2Fv%2Fstring_type)) ([How to use Pydantic model as graph state](https://langchain-ai.github.io/langgraph/how-tos/state-model/#:~:text=,integer)).
- This can significantly reduce bugs by catching mismatched assumptions early (for example, one agent returns a list where another expected a dict – a Pydantic model can flag that immediately).

## 7. Testing LangGraph Applications with Pytest (and Async Considerations)

Testing LLM-based applications can be challenging due to nondeterminism, but LangGraph’s structured approach makes it easier to isolate and verify components. Here are guidelines and patterns for testing LangGraph workflows with **pytest** (or any testing framework):

### 7.1 Unit Testing Tasks (Nodes)

**Unit test each `@task` function** whenever possible with regular function calls or by invoking them with stubbed data. Since tasks are just Python functions (possibly wrapping API calls), you can call the underlying function logic directly (if not using any LangGraph-specific magic inside). For tasks that call external services (like OpenAI API or tools), use mocking to simulate responses:
- Use `unittest.mock` or pytest monkeypatch to replace the actual API call with a deterministic function that returns a known output.
- For example, if `call_model` task calls `model.invoke()`, monkeypatch `model.invoke` to return a preset object mimicking an LLM response (e.g., a fake object with `.tool_calls` etc.).

This way, you can test that given certain inputs (prompt messages), the task returns the structure you expect, without calling the real API (saving cost and time).

**Assert validations**: If you use Pydantic models or have explicit checks, write tests that pass invalid data to ensure your task raises an error or handles it gracefully.

### 7.2 Testing Workflow Logic (Entrypoints)

The entrypoint function contains the control flow, which is crucial to test (especially for multi-step or multi-agent logic). However, you don’t want the real LLM or long tools running in each test. The strategy is to **mock tasks when testing the entrypoint**:
- Monkeypatch the `@task` functions used inside the entrypoint to dummy implementations that return predetermined values quickly.
- Alternatively, if the tasks are defined in the same module, you might call `task.invoke = lambda *args, **kwargs: StubFuture(result_value)` to bypass actual execution.

A pattern suggested by community experts ([How to write tests for Langgraph Workflows · langchain-ai langgraph · Discussion #633 · GitHub](https://github.com/langchain-ai/langgraph/discussions/633#:~:text=My%20preferred%20option%20to%20test,graphs%20would%20be%20the%20following)) ([How to write tests for Langgraph Workflows · langchain-ai langgraph · Discussion #633 · GitHub](https://github.com/langchain-ai/langgraph/discussions/633#:~:text=,call%20the%20actual%20model%20but)):
- **Unit test routing logic**: Provide fake implementations for tasks but keep the entrypoint’s branching and looping intact. For example, in a multi-agent workflow, make a fake `ask_travel_agent` that returns a handoff signal, and a fake `ask_hotel_agent` that returns a final answer. Then assert that your entrypoint loop indeed switched to hotel agent and returned the expected final answer.
- This ensures that the **flow** (edges of your “graph”) works as intended for various scenarios (e.g., one where no handoff occurs, one where multiple handoffs occur, etc.).

To mock a LangGraph task’s behavior, you might set the function’s `.result` method via monkeypatch. Another simpler way: since tasks can be called like normal Python (if you ignore the future part), you could monkeypatch the entire function. For instance:

```python
# inside test function
from my_app import ask_travel_agent, ask_hotel_agent, multi_agent_workflow

def fake_ask_travel_agent(query):
    class DummyResp: pass
    # simulate an agent response that triggers a handoff
    DummyResp.tool_calls = []  # if expecting attribute
    return "Please transfer to hotel agent for details."
# monkeypatch the task function to our fake (need to replace .result usage carefully if any)
monkeypatch.setattr(my_app, "ask_travel_agent", lambda q: type("F", (), {"result": lambda self=None: fake_ask_travel_agent(q)})() )
```

The above is a bit involved; an alternative is designing your tasks to have injectable dependencies so you can directly call a fake version. Or, structure your code so that the core logic (like deciding next agent) is in a pure function that you can call with synthetic inputs.

**End-to-End vs Unit**: Not everything can be unit tested easily (LLM behavior might be too dynamic to mock fully). Identify critical paths and create integration tests for them:
- **Integration test with cached LLM outputs**: One approach is to run the workflow once with the real LLM (maybe in a dev environment) and record the outputs. Store these outputs (could serialize the LangGraph events or final results) in a test fixture. In tests, instead of calling the LLM API, load these cached results. This is similar to using VCR cassettes or snapshots. LangGraph’s LangSmith integration or callback traces could assist in capturing these results.
- Alternatively, manually craft likely LLM outputs for given prompts and use those in tests.

The goal is to make tests deterministic:
- Set LLM `temperature=0` to reduce randomness.
- Provide fixed random seeds if any stochastic processes.
- Use smaller or more predictable models for tests (maybe a local model or a dummy chain that echoes inputs).

### 7.3 Testing State and Memory

For workflows with memory, write tests to ensure state carries over:
- Simulate a sequence of calls as a user would do in conversation. For example, call the entrypoint twice with the same thread_id and different user inputs, and assert that the second response differs based on the first input (indicating context was remembered).
- Check that the state saved (`previous` or `store` content) is as expected. You can directly inspect the `MemorySaver` contents if accessible, or use the return of `entrypoint.final` (it often returns the final value, but if you need to ensure `save` had correct content, you might need to adjust design or use a custom checkpointer that exposes its data for tests).
- If using long-term memory store, populate it with known data and see if the workflow retrieves it. Also test updates: run the entrypoint that modifies the store, then query the store after to see if changes persisted.

Because LangGraph’s memory system relies on the `config` with thread_id, be sure to pass a consistent config in your test calls. Example:

```python
config = {"configurable": {"thread_id": "test-session"}}
result1 = agent.invoke("Hello", config=config)
result2 = agent.invoke("Hi again", config=config)
assert "Hello" in some_form_of(result2)  # pseudo-check that context used
```

### 7.4 Asynchronous and Streaming Tests

LangGraph supports async execution, and Chainlit itself encourages async handlers. When writing tests for async code:
- Use `pytest.mark.asyncio` to define asynchronous test functions that can `await` calls.
- If your entrypoint or tasks are async (i.e., defined with `async def`), you can `await entrypoint.invoke_async(...)` or, if using stream, collect results like `results = [chunk async for chunk in entrypoint.astream(input)]`.
- Test that streaming yields the expected sequence of events. For example, a generation might stream tokens – you could assert that a certain token eventually appears, or that the final chunk matches the complete output.
- Test interrupt flows by simulating the user response. For instance, run the workflow until interrupt (maybe by calling `.stream()` and breaking when interrupt is hit), then resume with a dummy Command, and verify the final result.

Because handling concurrency can be tricky in tests, you might choose to run synchronous versions of code for simplicity. LangGraph tasks running in parallel (as futures) still can be tested in a synchronous manner by controlling when you call `.result()`. If you truly spawn parallel tasks (like making multiple API calls), you could mock those calls to deliberately include a delay or order and ensure the framework handles it.

### 7.5 Example: Pytest for a ReAct Agent

Suppose we have the `agent` entrypoint from section 3. We want to test that if the LLM outputs a tool call, our agent actually returns a final answer that includes the tool result:
```python
def test_agent_tool_usage(monkeypatch):
    from my_app import agent, call_model, call_tool

    # Monkeypatch call_model to return a dummy response with a tool call
    class DummyLLMResponse:
        def __init__(self, content, tool_name=None, tool_args=None):
            self.content = content
            self.tool_calls = []
            if tool_name:
                # Simulate a LangChain LLMResult with a tool call
                self.tool_calls = [ {"id": "123", "name": tool_name, "args": tool_args or {}} ]
    def fake_call_model(messages):
        # Always ask for weather via tool regardless of input
        return DummyLLMResponse("I should use a tool", tool_name="get_weather", tool_args={"location": "Paris"})
    monkeypatch.setattr(call_model, "result", lambda self=None: fake_call_model(None))

    # Monkeypatch call_tool to return a dummy ToolMessage
    def fake_call_tool(tool_call):
        # Simulate tool output
        return ToolMessage(content="Sunny in Paris.", tool_call_id=tool_call["id"])
    monkeypatch.setattr(call_tool, "result", lambda self=None, tool_call=None: fake_call_tool({"id":"123"}))

    # Now when we invoke agent, it should go through one loop of tool usage
    user_msg = [{"role": "user", "content": "What's the weather in Paris?"}]
    response = agent.invoke(user_msg, config={"configurable": {"thread_id": "test"}})
    assert "Sunny in Paris" in str(response)
```

The above test patches:
- `call_model.result` to bypass actual LLM and produce a tool call for `get_weather`.
- `call_tool.result` to bypass actual tool execution and return a preset observation.
Then it calls the `agent` entrypoint with a user query. The assertion checks that the final answer includes the tool’s information ("Sunny in Paris"), meaning the loop worked.

This approach isolates the logic of the loop from the unpredictable LLM/tool and makes the test deterministic.

### 7.6 Using LangSmith or Tracing for Testing

LangChain/LangGraph’s LangSmith (or tracing callbacks) can help in testing by capturing intermediate steps. You might use it in a test to ensure certain sequence:
- After running the agent, inspect the trace to confirm the order: e.g., user message -> LLM call -> tool call -> LLM call -> final.
- LangSmith also has a pytest integration ([Test a ReAct agent with Pytest/Vitest and LangSmith](https://docs.smith.langchain.com/evaluation/tutorials/testing#:~:text=Test%20a%20ReAct%20agent%20with,to%20evaluate%20your%20LLM%20application)) ([How to Properly Test RAG Agents in LangChain/LangGraph? - Reddit](https://www.reddit.com/r/LangChain/comments/1izqrhz/how_to_properly_test_rag_agents_in/#:~:text=How%20to%20Properly%20Test%20RAG,don%27t%20break%20the%20expected)) to treat evaluation datasets as tests. If you have a set of input prompts and expected outputs, you can use that to automatically validate your agent’s responses and even measure performance over time.

**Performance Tests**: If concurrency is used, you might want to test that it indeed runs faster or doesn’t block. This is more advanced, but you could time the execution with and without parallel tasks (using `pytest` markers or just print debug info).

**Error Handling**: Write tests for failure modes too. For example, if a tool fails (raises exception), does your workflow catch it and respond gracefully? You can simulate exceptions by monkeypatching a tool to raise, and asserting that the agent output contains an apology or error message.

In summary, for testing:
- **Break it down**: test tasks and small functions in isolation whenever you can.
- **Control randomness**: mock external calls, set seeds, use deterministic models.
- **Simulate flows**: test the branching logic by substituting dummy data for each branch.
- **Leverage LangGraph features**: its structured nature means you can often replay a fixed sequence and get the same result. Use that to your advantage in tests, possibly recording a “golden run” and comparing future runs to it (regression testing).
- **Pytest fixtures**: use fixtures to set up any heavy components (like instantiate an agent once for multiple tests) and to load test data (like sample prompts or fake API replies).
- **Async tests**: use `pytest.mark.asyncio` for any coroutine tests; test streaming by accumulating outputs from `entrypoint.stream()`.

By following these practices, you’ll gain confidence in the reliability of your LangGraph application. Remember that even though LLM outputs can change, your surrounding logic should be robust – testing helps ensure that, for example, even if the phrasing changes, the agent still calls the right tool or preserves the state. It’s about testing the **framework of the conversation** more than the exact natural language.

## 8. Asynchronous LangGraph Functional API Usage (Parallelism & Streaming)

LangGraph’s Functional API and Chainlit are both asynchronous-friendly. Writing workflows that execute concurrently and stream results can significantly improve performance and user experience (e.g., parallel API calls or token-by-token streaming of LLM responses). This section provides guidelines for using async features.

### 8.1 Concurrent Task Execution

As hinted earlier, you can invoke multiple tasks “at the same time” and then wait for their results, allowing parallel execution ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=Parallel%20execution%C2%B6)) ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=%40entrypoint%28checkpointer%3Dcheckpointer%29%20def%20graph%28numbers%3A%20list%5Bint%5D%29%20,result%28%29%20for%20f%20in%20futures)). This is particularly useful for IO-bound operations like calling multiple external APIs or tools.

**Pattern**: Instead of doing:
```python
res1 = task1(param).result()
res2 = task2(param).result()
```
which runs them sequentially, you can do:
```python
future1 = task1(param)  # don't call .result() yet
future2 = task2(param)
res1 = future1.result()
res2 = future2.result()
```
Between starting `future1` and calling `future1.result()`, the actual execution of `task1` can happen asynchronously. If `task2` doesn’t depend on `task1`, starting it before waiting for `task1` means both can run in parallel ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=Tasks%20can%20be%20executed%20in,calling%20APIs%20for%20LLMs)) ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=def%20graph%28numbers%3A%20list%5Bint%5D%29%20,result%28%29%20for%20f%20in%20futures)). Under the hood, LangGraph likely uses threads or event loops to execute tasks concurrently (ensuring thread-safe if needed for I/O).

**Example**: Suppose your agent, at some point, needs to fetch data from two different APIs to answer a question (like weather and news). You could do:

```python
@task
def fetch_weather(city: str) -> str: ...
@task
def fetch_news(topic: str) -> str: ...

@entrypoint()
def get_info(city: str, topic: str):
    # kick off both API calls concurrently
    weather_future = fetch_weather(city)
    news_future = fetch_news(topic)
    # now wait for both to finish
    weather = weather_future.result()
    news = news_future.result()
    combined = f"Weather: {weather}; News: {news}"
    return combined
```

This will likely be faster than calling one then the other, because while one API is waiting, the other can proceed.

**Asynchronous `async def` tasks**: If a task itself is defined as `async def`, LangGraph’s `@task` decorator should detect that and allow awaiting it. For instance:

```python
@task
async def call_llm_async(messages) -> LLMResult:
    result = await async_llm.generate(messages)  # assuming async llm client
    return result
```

You would still use it similarly (if called within an entrypoint, you might do `resp_future = call_llm_async(msgs)` then later `resp = resp_future.result()` – the `result()` call would internally await the coroutine).

**Avoiding common pitfalls**:
- Make sure tasks truly don’t depend on each other when running in parallel. If there is a dependency, you must .result() the first before starting the second (or else risk using uninitialized data).
- The order of `.result()` calls can be different from invocation order. You could, for example, start 5 tasks, then iterate `for fut in futures: results.append(fut.result())`. This will effectively wait for the slowest at each step, but they were all launched already.
- There’s no built-in mechanism to get results as they complete (like `asyncio.as_completed`) via LangGraph’s futures, but you can always design your own awaitable tasks if needed outside of LangGraph.

### 8.2 Streaming Outputs

LangGraph supports streaming in multiple ways:
- **Token streaming from LLMs**: If you use a streaming-enabled LLM (like OpenAI’s API with `stream=True`), LangChain can yield tokens incrementally. LangGraph will capture those as part of the run’s events.
- **Custom streaming via `StreamWriter`**: The Functional API provides a `StreamWriter` type that you can include in an entrypoint to send arbitrary data to a `"custom"` stream ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=Streaming%20custom%20data%C2%B6)) ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=from%20langgraph,types%20import%20StreamWriter)). For example, your workflow might produce intermediate progress messages or data that you want to stream to the frontend before the final result.

Using `.stream()` when invoking an entrypoint will yield a sequence of `(stream, data)` tuples or events. For instance:
```python
for event in my_entrypoint.stream(input, stream_mode=["custom", "updates"]):
    print(event)
```
You might see events like:
- `('updates', {'step1': 'partial result'})` indicating an update from a task ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=API%20Reference%3A%20MemorySaver%20%20,80)).
- `('custom', 'some message')` from your custom writer calls.
- `('interrupt', ...)` if an interrupt happened (as discussed).
- Final `'updates'` with the final result.

In Chainlit, you can forward these streamed events to the UI. Chainlit natively supports token streaming: if you use `cl.Message(content=..., stream=True)` and keep sending tokens, the UI will animate them. With LangGraph, if you intercept token-by-token events, you can call `await cl.Message(content=token, stream=True).send()` for each token.

**Integrating Streaming in Chainlit**:
- If the LangGraph entrypoint returns an `StreamingIterable` or you use `entrypoint.stream`, then in your Chainlit handler, iterate over it and send messages as they come.
- Chainlit’s `@cl.on_message` can be async, so you can `async for chunk in agent.astream(user_input, config=...)` as well. Or use synchronous `.stream` in a separate thread if needed (Chainlit allows sync usage but encourages async).

**Example**: Streaming a response:
```python
@cl.on_message
async def on_message(msg: str):
    # Start the agent in streaming mode for both LLM token stream and custom messages
    for stream, data in agent.stream(msg, stream_mode=["tokens", "custom"], config={"configurable": {"thread_id": "xyz"}}):
        if stream == "tokens":
            # Suppose data is a single token or chunk of text
            await cl.Message(content=data, author="assistant", stream=True).send()
        elif stream == "custom":
            # Maybe custom stream used for debugging or other info
            print(f"Custom event: {data}")
        elif stream == "interrupt":
            # (Pseudo-code) handle interrupt if it appears in stream events
            prompt = data['value'] if isinstance(data, dict) else str(data)
            user_answer = await cl.AskUserMessage(content=prompt).send()
            # resume logic as earlier...
```
Chainlit will ensure the UI displays tokens in order. Once the loop ends and the final message is sent (with `stream=False` on the last one automatically when `cl.Message.send()` completes streaming), the user sees the full answer.

**Async Concurrency in Chainlit**: If you plan to have multiple users or multi-turn with overlapping tasks, remember that Chainlit’s `on_message` for different sessions run concurrently. The LangGraph `thread_id` keeps each session’s state separate, but heavy parallelism might still strain resources (e.g., many LLM calls at once). Use async features to not block the server loop, and consider rate limiting or queueing if needed.

### 8.3 Async Entrypoints and Tasks

You might wonder if you can make the entrypoint itself `async`. In Python, an `async def` entrypoint decorated with `@entrypoint` likely works, but typically you don’t `await` an entrypoint; you call `.invoke()` or `.stream()` on it. Under the hood, LangGraph might run it in an event loop. If you need to do asynchronous operations in the entrypoint, you can still call async tasks or other asyncio code by awaiting within tasks or using Python concurrency primitives.

**Streaming and Memory**: They work together – as you stream out tokens or intermediate results, the state is still being built. Only once the entrypoint finishes do we checkpoint the final state. So if a run is interrupted (e.g., error or manual stop), you might not have saved the latest state (unless using advanced “time travel” features of LangGraph).

**Error Handling in Async**: If any task raises (e.g., API fails), how to handle it? You can include try/except inside tasks or the entrypoint. Alternatively, LangGraph might allow a `retry` policy on tasks ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=Retry%20policy%C2%B6)) ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=%40task,attempts%20%2B%3D%201)). For example, you can specify `@task(retry=RetryPolicy(...))` to auto-retry certain exceptions. This is useful to handle flaky API calls in a workflow without crashing the whole run.

**Cancelling tasks**: If the user stops the Chainlit conversation mid-way, ideally you’d cancel the LangGraph run. Chainlit might not yet have a direct cancel mechanism, but you could implement a cooperative cancellation by checking some flag in long loops.

### 8.4 Example: Parallel Agents (Advanced Async)

Think of a scenario: you have multiple agents (e.g. 3 different specialists) and you want them to **work concurrently** on parts of a problem, then gather their answers. With LangGraph, you could spawn each agent as a task:

```python
@task
def agent1_task(question): ...
@task
def agent2_task(question): ...
@task
def agent3_task(question): ...

@entrypoint()
def multi_agent_concurrent(question: str):
    futures = [agent1_task(question), agent2_task(question), agent3_task(question)]
    results = [f.result() for f in futures]  # run all three in parallel
    # Combine or choose among results
    final = combine_answers(results)
    return final
```

This runs all three agents in parallel, cutting down latency significantly compared to sequential queries. The combine logic could simply concatenate answers or have another LLM decide the best. In testing, ensure that parallel execution doesn’t cause race conditions (each agent is separate so usually fine) and that resource usage is acceptable (3 simultaneous LLM calls).

**Performance note**: Python’s GIL means threads can’t run Python code truly in parallel, but I/O operations can overlap. If tasks are mostly I/O bound (calling external services), this approach is beneficial. If tasks are CPU-bound, consider using `asyncio` for I/O tasks or even multiprocessing for CPU-bound tasks (though that complicates LangGraph usage).

### 8.5 Monitoring Async Workflows

Using LangGraph’s observability tools (like LangSmith or built-in logging), you can monitor parallel tasks and streaming:
- Each `task` invocation likely logs when it starts and ends. By reviewing logs or traces, you can confirm that tasks overlapped in time.
- For streaming, ensure that partial outputs are as expected (no missing chunks, etc.). In case of anomalies, debug whether the issue is in the model streaming or in how events are forwarded.

In conclusion, the Functional API’s async capabilities allow you to build *efficient and responsive* LLM applications:
- Utilize parallel tasks to handle tool calls or subtasks concurrently ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=Tasks%20can%20be%20executed%20in,calling%20APIs%20for%20LLMs)) ([Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/#:~:text=def%20graph%28numbers%3A%20list%5Bint%5D%29%20,result%28%29%20for%20f%20in%20futures)), improving throughput.
- Embrace streaming to provide users faster feedback (partial answers or progress indicators) rather than waiting for the entire response.
- With Chainlit, tie these into a UI that updates in real time, giving the feel of a live conversation or process.

By combining all the above – multi-agent design, tool use, memory, Pydantic for robust data, testing for reliability, and async for performance – you can implement complex AI workflows that are **modular, maintainable, and user-friendly**. LangGraph’s Functional API and Chainlit together offer a powerful stack for building next-generation AI applications.


