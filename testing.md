Thanks for the details! I'll put together a guide on handling LangGraph context for testing, specifically focusing on Python, both unit and integration tests, and using the LangGraph Functional API (`@task`, `@command`). The guide will also be tailored for Pytest, ensuring you have clear steps for setting up and managing LangGraph context within your tests. I'll let you know when it's ready!

# Handling LangGraph Context in Pytest: A Comprehensive Testing Guide

## Introduction  
LangGraph is built on a **runnable context** architecture. This means that tasks, tools, and workflows expect to run within a managed context that provides configuration and state. If you call LangGraph functions outside this context (for example, directly invoking a `@task` function in a test), you may encounter runtime errors such as **"RuntimeError: Called get_config outside of a runnable context"**  for its tasks and commands to function properly**, and skipping that context in tests leads to missing config/state and runtime failures.

**Why a runnable context?** LangGraph uses context variables to propagate configs (like `thread_id`, model selection, etc.) and to enable features like persistence, memory, and tracing. When you run a LangGraph workflow via `.invoke()` or similar, it sets up this context so that calls to `get_config()` or `ensure_config()` return the correct settings. If you try to execute the code outside that flow, these functions have no context to draw from and raise errors. This was especially problematic in older Python versions where automatic context propagation is limited; for example, in Python 3.10 async runs require manually passing the config to subcalls . The takeaway is that **your tests must simulate or provide the LangGraph runtime context** to avoid such errors.

## Unit Testing LangGraph Functions  
**Goal:** Test individual `@task` or `@command` functions in isolation, ensuring their logic works as expected without running the entire graph.

**1. Testing Tasks/Commands in Isolation:** LangGraph’s `@task` decorator turns a function into a *future-like runnable*. Calling a `@task` function returns an object whose result you can obtain via `.result()` or by awaiting it. For pure functions (those that don’t depend on the LangGraph context), you can simply call the task and resolve its result in a test. For example: 


from langgraph.func import task

@task
def add_numbers(a: int, b: int) -> int:
    return a + b

def test_add_numbers():
    future = add_numbers(2, 3)        # Call the task (returns a future-like object)
    assert future.result() == 5       # Resolve the result and verify
 

In this simple case, no special context is needed since `add_numbers` doesn’t use any runtime config. The task executes normally and we verify the outcome. This follows the advice of unit-testing each node’s logic independently .

**2. Simulating Runnable Context with Mock Config:** If the function *does* rely on runtime config or state (e.g. it calls `ensure_config()` or uses an injected `config` parameter), you need to simulate that context in your unit test. One approach is to provide a **mock `RunnableConfig`**. A `RunnableConfig` in LangGraph is essentially a dictionary of settings passed at runtime . For unit tests, you can create a dummy config dict and force your task to use it. There are a few ways to do this: 

- **Inject `config` via function parameters**: If you declare a parameter of type `RunnableConfig` in your task function signature, LangGraph will inject the runtime config there when running in a workflow. You can leverage this for testing by calling the function directly with a dummy config. For example: 

  
  from langchain_core.runnables.config import RunnableConfig
  from langgraph.func import task

  @task
  def greet(name: str, config: RunnableConfig) -> str:
      # Use the injected config to determine greeting
      greeting = config["configurable"].get("greeting", "Hello")
      return f"{greeting}, {name}!"
   

  In production, this task would get `config` automatically. In a test, we can call it with our own config:

  
  def test_greet_custom():
      dummy_config = {"configurable": {"greeting": "Hi"}}
      result = greet("Alice", config=dummy_config).result()
      assert result == "Hi, Alice!"
   

  Here we simulate the runnable context by providing a `dummy_config`. The task uses that instead of calling `ensure_config()`, avoiding the outside-context error. Defining tasks to accept a config param is a convenient way to make them testable, as shown in LangGraph’s own examples .

- **Monkeypatch `ensure_config`**: If modifying the function signature isn’t feasible, you can patch the LangGraph config getter in tests. For instance, if your task uses `from langgraph.config import ensure_config` internally, you can monkeypatch this to return a preset config. Using Pytest’s `monkeypatch` fixture: 

  
  from mymodule import fetch_user_flight_information  # a @task or @tool function
  def test_fetch_flight_info(monkeypatch):
      # Simulate a runnable context by patching ensure_config to return desired config
      monkeypatch.setattr("mymodule.ensure_config", lambda: {"configurable": {"passenger_id": "ABC123"}})
      result = fetch_user_flight_information().result()  # call the task
      # Now the function will find passenger_id in config and not raise an error
      assert isinstance(result, list)
      # (add assertions about result contents here)
   

  In this example, `fetch_user_flight_information` was likely decorated (e.g. with `@tool` or `@task`) and uses `ensure_config()` to get a `passenger_id`. By patching `ensure_config` to return a fake config (with `"passenger_id": "ABC123"`), we mimic the runnable context. This prevents `RuntimeError` and allows us to verify the function’s logic. **Note:** This approach tests the function’s logic by bypassing LangGraph’s injection — effectively treating it as a normal function once we supply the needed config.

**3. Verifying Logic with Pytest Assertions:** With the above setup, you can use standard Pytest assertions to check that the function produces expected outcomes. For pure tasks, assert on the return value. For tasks that modify state or have side effects, you might need to inspect global state or outputs. When designing tasks, try to return values or state changes that can be asserted in tests rather than relying purely on side effects (this makes testing easier).

**4. Example – Unit Test a LangGraph Task:** Consider a task that formats a greeting and uses a runtime config setting:


# In module myapp.py
from langgraph.func import task
from langgraph.config import ensure_config

@task
def format_greeting(user: str) -> str:
    # This task uses a config value to customize the greeting
    config = ensure_config()  # Will fetch RunnableConfig from context
    greeting = config.get("configurable", {}).get("greeting", "Hello")
    return f"{greeting}, {user}."


To unit test this, we must provide the `greeting` config. We can either refactor to accept `config` as an arg, or simply patch `ensure_config` in the test:


# In test_myapp.py
import myapp
def test_format_greeting(monkeypatch):
    # Arrange: set up a dummy config for the context
    dummy_conf = {"configurable": {"greeting": "Hi"}}
    monkeypatch.setattr(myapp, "ensure_config", lambda: dummy_conf)
    # Act: call the task function
    future = myapp.format_greeting("LangGraph")
    result = future.result()  # resolve the future to get actual return
    # Assert: verify the greeting used the config
    assert result == "Hi, LangGraph."


This test ensures the task’s logic (using the config value) works, **without needing to run an entire workflow**. It isolates the unit’s behavior, which is a recommended practice . By simulating the context, we avoided the `RuntimeError` and validated the function’s output.

## Integration Testing LangGraph Graphs  
Unit tests cover individual pieces, but **integration tests** ensure that multiple LangGraph components work together as a whole. In integration tests, you'll execute actual LangGraph workflows (graphs or entrypoints) end-to-end within Pytest.

**1. Setting up Graph Execution in Tests:** To test a LangGraph workflow, first construct the graph or functional workflow as you would in your application, including any checkpointer or store if needed. For example, suppose we have an `@entrypoint` workflow that uses some tasks:


# myapp_workflow.py
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver

@task
def double(x: int) -> int:
    return x * 2

@task
def increment(x: int) -> int:
    return x + 1

# Use an in-memory checkpointer to enable state persistence (optional for simple flows)
checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def math_workflow(inputs: dict) -> dict:
    """Entrypoint that doubles a number and then increments it."""
    y = double(inputs["number"]).result()      # first task
    z = increment(y).result()                 # second task
    return {"result": z}


In a Pytest, we can execute this workflow using its `.invoke()` method:


# test_myapp_workflow.py
from myapp_workflow import math_workflow

def test_math_workflow_end_to_end():
    # Provide input and a config (including a thread_id for reproducibility)
    config = {"configurable": {"thread_id": "test-run-1"}}
    output = math_workflow.invoke({"number": 3}, config=config)
    assert output == {"result": 7}


Here we’re calling the compiled workflow’s `invoke` method with an input dict. The config includes a `thread_id` which identifies the run thread – using a fixed ID in tests can help reuse or inspect stored state, but you should ensure uniqueness if state should not carry over between tests. The `.invoke` call runs the entire workflow in the LangGraph runtime (setting up context, executing tasks in order). We then assert that the final output matches the expected result. This is a straightforward integration test covering multiple tasks together.

**2. Invoking Workflows and Handling State:** In integration tests, you might run workflows that maintain state across invocations or use memory. For example, if you want to test that a conversation agent carries context between turns, you would call `.invoke` multiple times with the same `thread_id` and verify the state evolves. Make sure to initialize or reset any persistent state (e.g., using a fresh `MemorySaver` or unique thread IDs per test) to avoid bleed-over between tests.

If your workflow involves human-in-the-loop interrupts, integration testing requires simulating the resume step. LangGraph provides a `Command` object for resuming an interrupted workflow. For instance, if `math_workflow` above had an `interrupt()` call waiting for input, the first `.invoke` would return an `Interrupt` or you would capture it via streaming. To test it end-to-end, you would then call something like: 


from langgraph.types import Command

# ... after capturing an interrupt that expects a resume value:
resume_val = True  # example resume value for approval in an HIL scenario
final_output = math_workflow.invoke(Command(resume=resume_val), config=config)


Using `Command(resume=...)` tells LangGraph to continue the workflow with the given input, bypassing already completed tasks . In Pytest, you can simulate a user response by directly supplying a `Command` object on the next invoke or stream call. This allows testing of workflows with dynamic pauses.

**3. Example – Integration Test for a LangGraph Workflow:** Let’s integrate the above ideas with a more realistic scenario. Imagine a LangGraph workflow that uses an LLM and a tool (search function) in sequence. In the application, you’d have something like:


# app_agent.py
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain.chat_models import ChatOpenAI

# Define a simple tool
@tool
def search_weather(city: str) -> str:
    """Dummy tool: returns weather for a city."""
    if city.lower() == "san francisco":
        return "Foggy 60°F"
    return "Sunny 90°F"

tools = [search_weather]
llm = ChatOpenAI(temperature=0)  # deterministic LLM for testing
agent_app = create_react_agent(llm, tools, checkpointer=MemorySaver())


In this agent, the LLM will decide whether to call the `search_weather` tool. For an integration test, calling the agent with a user query and verifying the final state or answer is the goal. However, hitting the real LLM (even with temperature=0) is slow and not deterministic beyond the fixed randomness. **A better approach is to stub out external dependencies** like the LLM. We can inject a fake LLM or monkeypatch the LLM’s invoke method to return a predetermined response. Similarly, if the tool were complex or made external calls, we’d stub it too. The LangGraph design often allows injecting dependencies for exactly this reason. In our example, `create_react_agent` let us supply our `llm` and `tools`. In tests, we can replace them with mocks.

Using Pytest, we can write:


# test_app_agent.py
from langchain.schema import AIMessage
from app_agent import create_react_agent, search_weather

def test_agent_weather_tool(monkeypatch):
    # 1. Set up fake LLM that always produces a specific response (to force a tool call or not).
    fake_llm = lambda messages: AIMessage(content="Tool: search_weather")  # instruct agent to call tool
    monkeypatch.setattr("app_agent.llm", fake_llm)  # patch the llm used in agent
    
    # 2. (Optional) Patch the tool if we want a custom behavior or to spy on it.
    # In this case, our search_weather is simple enough, but we could do:
    # monkeypatch.setattr("app_agent.search_weather", lambda city: "TestWeather")
    
    # 3. Create the agent with the patched components
    agent_app = create_react_agent(fake_llm, [search_weather], checkpointer=None)
    
    # 4. Invoke the agent workflow with a test input
    result_state = agent_app.invoke({"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]})
    
    # 5. Verify that the final answer contains the tool result (foggy weather).
    final_message = result_state["messages"][-1]["content"]
    assert "Foggy 60°F" in final_message


In this integration test, we: (a) monkeypatch the LLM to control its output (ensuring the agent's behavior is predictable), (b) optionally patch the tool (not done above, but we could if needed to isolate from real logic), (c) invoke the LangGraph agent, and (d) assert that the final agent response includes expected content from the tool. This approach aligns with recommended practices: **use mocks or fakes for external services to keep tests deterministic and fast** . We effectively turned the possibly nondeterministic LLM call into a predictable function, making our test repeatable.

**4. Managing State and Configuration in Integration Tests:** Each integration test should be self-contained. If your LangGraph workflows use persistent state (via a checkpointer), consider using an in-memory checkpointer (like `MemorySaver`) for tests, and reset it or use unique `thread_id`s per test so that runs don’t interfere. Always pass a `config` when invoking if your workflow or tasks expect one. For example, if your graph relies on config values (like the model name in a config), supply that in the `.invoke(...)` call. The LangChain/LangGraph docs note that `invoke` (and related methods) accept an optional config dict as the second argument  – use this to your advantage in tests to set things like `{"configurable": {...}}` or callbacks.

**5. Covering Various Scenarios:** Write integration tests for both **happy path** and **edge cases** in your workflow. For instance, test a path where a condition in your `@entrypoint` takes one branch vs another (you can force branches by controlling task outputs via monkeypatching). If your graph has error handling (retries, etc.), you can simulate a failure in a task (e.g., monkeypatch a task to raise an exception) and verify the workflow handles it gracefully. By setting up controlled inputs and stubbed components, you can simulate error paths without flakiness. 

Remember that integration tests run the actual LangGraph engine – they will be slower than unit tests. You might not want to test every minor logic combination at this level. Instead, focus on overall correctness of the assembled workflow and do more granular checks in unit tests for the pieces.

## Best Practices for Testing LangGraph Workflows  
When testing LangGraph-based applications, keep in mind these best practices to make your tests effective and maintainable:

- **Design for Testability:** Where possible, write your LangGraph tasks and tools in a way that logic can be tested outside the framework. For example, encapsulate complex computations in plain Python functions or methods, and have your `@task` functions call them. This lets you test the logic with simple function calls, bypassing the LangGraph runtime entirely for unit tests. Use LangGraph decorators (`@task`, `@tool`, `@entrypoint`) as thin wrappers around testable logic. This separation means fewer tests need to deal with context issues at all.

- **Use RunnableConfig Injection Instead of Global Calls:** Instead of calling `ensure_config()` inside your tasks/tools, consider accepting a `config: RunnableConfig` parameter (as shown earlier). This makes it easier to supply config in tests. It also makes your code clearer about its dependencies. The LangChain docs demonstrate that adding a `config` parameter to your tool or task allows the active config to be passed in automatically ` at all in tests – you’ll already have the config.

- **Mock External Dependencies:** LangGraph workflows often integrate external services (LLM APIs, databases, web requests). **Do not call real APIs in unit tests.** Instead, **mock them out**. Use Pytest’s monkeypatch or the `unittest.mock` library to replace LLM calls with fake functions that return known outputs . The idea is to control randomness and external variability in tests.

- **Leverage Low-Temperature and Determinism:** If you *must* hit an LLM in a test (for integration coverage), use parameters that minimize randomness. Set temperature to 0 and top_p to 0 (or 1) so the model is deterministic given the same prompt. Ensure the prompt is fixed. This reduces flaky tests. However, even with deterministic settings, external LLMs can sometimes change outputs over time or with API changes, so prefer mocking when possible. You could also record a known-good response and have the mock return that (approach often called *snapshot testing* or *cached responses*).

- **Isolate Tests:** Each test function should set up and tear down cleanly. If using a global `MemorySaver` or persistent store for LangGraph, reset it or use new instances in tests. Avoid reusing `thread_id`s across unrelated tests if using the same checkpointer, as LangGraph will otherwise carry over state from one run to the next. If you want to test persistence, do so deliberately (e.g., call a workflow twice in one test with the same `thread_id` and assert that state persisted). Otherwise, keep runs independent.

- **Use Integration Tests Sparingly and Strategically:** Integration tests are valuable but can be slower and more complex. Use them to cover the interactions between tasks and overall workflow logic – for example, that the right tool is called in an agent, or the overall output for a scenario is correct. You don’t need to cover every internal branch with an integration test if your unit tests already ensure individual tasks behave correctly for those branches. A common strategy is to have a few end-to-end scenarios tested with real (or high-fidelity stubbed) components – just to ensure the wiring is correct – while using unit tests to exhaustively test edge conditions in isolation .

- **Employ Caching for Expensive Calls:** For integration tests that involve expensive calls (like LLM generations), you can implement a caching layer. One approach is to record the outputs of those calls on the first run and save them (to a file or in-memory structure), then have subsequent test runs load those outputs instead of calling the API. This way, your integration tests run fast and don’t depend on external services except on a cache miss. As suggested in the LangGraph discussions, you can **cache model outputs** so that a test rerun uses stored answers instead of hitting the model, greatly speeding up tests and avoiding fluctuations . If you change your prompt or logic, you might refresh the cache, but during regular development the cache makes tests reliable and quick.

- **Debugging Test Failures:** If a LangGraph test fails, first check the error message. A `RuntimeError` about context usually means you forgot to provide a config or run the code via `.invoke()`. Ensure that any function decorated with `@task` or `@entrypoint` is not being called like a normal function (unless you patched its context). Always use `.invoke()`/`.result()` in tests to execute the graph or task. For logic errors, it can be useful to utilize LangChain’s debugging tools even in tests – for example, enabling LangSmith tracing to see what happened inside the workflow, or printing intermediate state from your tasks. You can also break up the workflow and test sub-parts to isolate where the issue is. Since LangGraph tasks are Python functions, you can unit test them individually to pinpoint logic issues before they even run in the workflow.

- **When to Use Real Execution vs Mocking:** Use real execution (actual `.invoke()` with real LLM calls) **only for high-level integration tests** that simulate real usage, and even then preferably with deterministic settings or recorded outputs. Use mocking for **unit tests and most integration tests** to eliminate external unpredictability. This balanced approach gives you confidence in the system’s integration (with a couple of full runs) without making the entire test suite slow or flaky. As one expert recommended, a **test pyramid** for LangGraph might include: many unit tests for nodes, some tests for graph logic with stubbed nodes, and a few end-to-end tests with real or cached outputs . This ensures broad coverage with efficiency.

## Example Code Snippets  

Below are simplified examples illustrating LangGraph testing techniques with Pytest:

### Example 1: Unit Testing a LangGraph Task

**LangGraph Function (`@task`)** – We define a task that depends on a config value (for demonstration):


# file: mytasks.py
from langgraph.func import task
from langgraph.config import ensure_config

@task
def compute_discount(price: float) -> float:
    """
    Apply a discount rate from config to the given price.
    Expects config.configurable['discount'] to be a percentage (0-100).
    """
    config = ensure_config()  # get the runtime config (must be in context)
    discount_pct = config.get("configurable", {}).get("discount", 0)
    return price * (1 - discount_pct/100.0)


If we call `compute_discount(100.0)` normally, it would try to fetch config and error out if none. So our test will provide a context.

**Pytest Unit Test**:


# file: test_mytasks.py
import mytasks

def test_compute_discount(monkeypatch):
    # Arrange: monkeypatch ensure_config to simulate context with a 10% discount
    monkeypatch.setattr(mytasks, "ensure_config", lambda: {"configurable": {"discount": 10}})
    # Act: call the task function
    future = mytasks.compute_discount(200.0)
    result = future.result()  # triggers execution of the task
    # Assert: 10% of 200 is 20, so result should be 180.0
    assert result == 180.0


**Explanation:** We patched `ensure_config` to return `{"discount": 10}` within this test. When `compute_discount` runs, it finds the discount rate and applies it. We then assert that the output is as expected. This isolates `compute_discount`’s logic. We did not need to compile a graph or use an entrypoint here – it’s a quick unit test. (In practice, if `compute_discount` were part of a larger workflow, that workflow would pass the config, but here we simulate it.)

### Example 2: Integration Testing a Workflow

**LangGraph Workflow:** Suppose we have an entrypoint that uses the above task and maybe another:


# file: myworkflow.py
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from mytasks import compute_discount

@task
def add_tax(price: float) -> float:
    """Simple task to add a fixed 5 currency units tax."""
    return price + 5.0

checkpointer = MemorySaver()  # enable persistence if needed
@entrypoint(checkpointer=checkpointer)
def pricing_workflow(data: dict) -> dict:
    """
    Workflow that computes discounted price and then adds tax.
    Expects input {'price': <float>}.
    """
    discounted = compute_discount(data["price"]).result()  # may use config for discount
    final_price = add_tax(discounted).result()
    return {"final_price": final_price}


**Pytest Integration Test:**


# file: test_myworkflow.py
from myworkflow import pricing_workflow

def test_pricing_workflow_integration():
    # We'll test with a discount config of 15%
    config = {"configurable": {"discount": 15}, "trace": False}
    input_data = {"price": 100.0}
    result = pricing_workflow.invoke(input_data, config=config)
    # final_price = (100 - 15%) + 5 tax = 85 + 5 = 90
    assert result == {"final_price": 90.0}


**Explanation:** We call `pricing_workflow.invoke` with a config specifying a 15% discount. This config will be available to the `compute_discount` task during execution (LangGraph injects it because we used an entrypoint and checkpointer). The workflow returns the final price after discount and tax, which we assert to be 90.0 for a 100 input. This test runs through the actual LangGraph runtime: it involves two tasks and uses the `MemorySaver` (though in this case we didn’t leverage persistence across calls). Note that we included `"trace": False` in config just to suppress any tracing in this context (assuming LangSmith tracing could be on by default; this is optional). The key is that by using `.invoke` with a proper config, **the tasks have their context and config, so no `RuntimeError` occurs**. We’re effectively testing the pieces together, ensuring that the discount from config is applied and then tax added.

### Example 3: End-to-End Test with Mocks (LLM Agent Example)

Consider a more complex scenario – a LangGraph agent that answers math questions by possibly using a calculator tool. We want to test the full agent logic without calling the real LLM API.

**Agent setup (LangGraph)**:


# file: math_agent.py
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain.chat_models import ChatOpenAI

@tool
def calculator(expression: str) -> str:
    """A simple calculator tool that evaluates a math expression given as string."""
    try:
        result = eval(expression)
    except Exception as e:
        result = str(e)
    return str(result)

tools = [calculator]
llm = ChatOpenAI(temperature=0)  # LLM for reasoning (we'll mock this in tests)
agent = create_react_agent(llm, tools, checkpointer=MemorySaver())


This agent is supposed to take a user question, decide if it needs to use the calculator, use it, and return an answer.

**Pytest Integration Test with Mock LLM:**


# file: test_math_agent.py
from langchain.schema import AIMessage, HumanMessage
from math_agent import agent, calculator

def test_math_agent_calculation(monkeypatch):
    # Prepare a fake LLM response to force a specific behavior.
    # Let's say the question is "What is 2+2?" and we expect the agent to call the calculator.
    user_question = "What is 2+2?"

    # Monkeypatch the agent's LLM to always respond with an instruction to use the calculator tool.
    fake_chain_of_thought = AIMessage(content="Thought: I should use the calculator.\nAction: calculator\nAction Input: 2+2")
    monkeypatch.setattr(agent, "llm", lambda messages: fake_chain_of_thought)

    # Now invoke the agent with the user question
    result_state = agent.invoke({"messages": [HumanMessage(content=user_question)]},
                                config={"configurable": {"thread_id": "math-test-1"}})
    # The agent should have used the calculator tool and gotten "4" as result, so final message should contain "4".
    final_answer = result_state["messages"][-1].content
    assert "4" in final_answer
    assert final_answer.strip().startswith("4") or "4." in final_answer  # e.g., "4" or "4. The answer is four."


**Explanation:** We monkeypatch `agent.llm` to a lambda that ignores the prompt and returns a crafted `AIMessage` indicating the agent’s reasoning and tool use (this mimics an internal step where the LLM decides to use the calculator with input "2+2"). In a real agent run, LangGraph would feed the user question to the LLM, get a chain-of-thought and tool usage directive, then call the `calculator` tool. By faking that step, we ensure the agent will call `calculator` with "2+2". The `calculator` tool itself is just a Python `eval`, which will return "4". LangGraph then would likely feed that back into the LLM for a final answer (depending on agent logic), but since we have a simple fake, we assume after one step it returns the final answer as the content of `result_state["messages"][-1]`. We assert that "4" is present, meaning the agent ultimately provided the correct calculation. This test ran the *entire agent workflow* with our controlled LLM behavior, so it’s an end-to-end test of the agent’s decision logic and tool integration. Crucially, by controlling the LLM output, we avoided randomness and made the test deterministic and fast.

### Example 4: Testing a Workflow with Human-in-the-Loop (HIL)

For completeness, let's sketch how you might test a workflow that includes a human review step (using LangGraph’s `interrupt` and resume with `Command`). This is an advanced scenario:


# file: review_workflow.py
from langgraph.func import entrypoint, task
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

@task
def generate_draft(topic: str) -> str:
    return f"Draft content about {topic}"

checkpointer = MemorySaver()
@entrypoint(checkpointer=checkpointer)
def content_review_flow(topic: str) -> dict:
    draft = generate_draft(topic).result()
    # Pause for human approval
    approved = interrupt({"draft": draft, "action": "approve_or_reject"})
    # The workflow will pause here until resumed with a Command containing 'resume' value.
    return {"draft": draft, "approved": approved}


This workflow generates a draft and then waits for a human to approve or reject (the `approved` variable will be provided by the resume command).

**Pytest Integration Test for HIL Workflow:**


# file: test_review_workflow.py
from review_workflow import content_review_flow
from langgraph.types import Command

def test_content_review_flow_approval():
    config = {"configurable": {"thread_id": "review-123"}}
    # First, invoke the workflow up to the interrupt (stream to capture it)
    events = list(content_review_flow.stream("TestTopic", config=config))
    # The last event should be an Interrupt waiting for approval
    assert any(event.get("Interrupt") or event.get("interrupt") for event in events), "Expected an interrupt event"
    # Now simulate a human approving the draft:
    resume_val = True  # let's say True means approved
    result = content_review_flow.invoke(Command(resume=resume_val), config=config)
    # Verify the workflow output includes the draft and the approval status True
    assert result["draft"].startswith("Draft content about TestTopic")
    assert result["approved"] is True


**Explanation:** We used `stream()` to run the workflow until it pauses (the `interrupt` yields an event). In the events, we expect an `Interrupt` indicator. After that, we call `invoke()` with a `Command(resume=True)` to resume the flow as if a user approved the draft. Finally, we check that the output indicates the draft content and `approved: True`. In a real application, the `interrupt` would surface to a UI or API and wait for input, but in testing we bypass that by immediately supplying the `Command`. This demonstrates that even human-in-the-loop flows can be tested programmatically. The key is to use the same `thread_id` and checkpointer so that the second call resumes the first call’s state . We ensured that by using the same `config` for both calls. This test ensures that the workflow correctly handles pausing and resuming logic.



By following these guidelines and patterns, you can confidently test LangGraph functional workflows with Pytest. **In summary**, always run LangGraph code in a proper context (or simulate one) during tests, break down your testing strategy into unit and integration levels, use dummy configs and mocks to control the environment, and leverage LangGraph’s design (like `RunnableConfig` and `Command`) to drive your workflows in tests. This will help avoid runtime context errors and make your tests reliable, fast, and easy to understand. With these best practices, you can iterate on LangGraph applications while maintaining a robust test suite that catches regressions and ensures your complex AI workflows behave as expected. 

**Sources:**

1. LangGraph requires a runtime context for config – GitHub discussion, *LangGraph get_config error*   
2. LangChain RunnableConfig propagation (context requirement) – *LangChain Docs on RunnableConfig*   
3. Recommended testing strategy (unit test nodes, then flow, etc.) – LangGraph Discussion on testing workflows   
4. Using stateful mocks and controlling LLM randomness in tests – LangGraph Q&A (Tanzimabsar and others)   
5. Injecting config via function parameters – LangGraph configuration how-to guide   
6. LangGraph Functional API reference – *LangGraph Docs: Functional API* (for usage of `@task`, `@entrypoint`, and `Command` resume) 
