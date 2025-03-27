# Project Enhancement Roadmap

### Phase 1: Command-Based Agent Triggers
#### Goal: Allow `/roll`, `/search`, etc. to bypass normal decision agent flow

- **Steps:**
  1. **Add Chainlit Commands**
  
      ```python
      from chainlit import on_command

      @on_command("roll")
      async def cmd_roll(session):
          # Directly invoke dice_roll_agent with default parameters
          state = session.get("state")
          result = await dice_roll_agent(state)
          await session.send(result[0].content)

      @on_command("search")
      async def cmd_search(session):
          query = session.message.content.split("/search ")[1]
          state = session.get("state")
          state.messages.append(HumanMessage(content=query))  # Fake user input
          result = await web_search_agent(state)
          await session.send(result[0].content)
      ```

  2. **Update event_handlers.py**
     - Add command registration imports
     - Modify on_message() to handle commands first:
  
       ```python
       if message.content.startswith('/'):
           await chainlit.handle_commands(message)
           return
       ```

  3. **Modify src/agents/decision_agent.py**
     - Exclude command messages from normal processing:
  
     ```python
     user_input = next(...).content
     if user_input.startswith('/'):
         return []  # Bypass decision agent
     ```



                                                          Phase 2: Helper Workflow Implementation

                                                    Goal: Daily automation for todos and data summaries

Components:

- Todo Manager Agent
- Daily Data Summary Agent
- External Query Proxy Agent

Directory Structure:

```
helper/
└── helper_for_{date}/
    ├── todo.md
    └── daily_data_summary.md
```


                                                                   Implementation Steps:

1. Todo Manager Agent


from datetime import date
from langgraph.func import task

@task
async def todo_agent(state: ChatState):
    todays_dir = f"helper/helper_for_{date.today()}/"
    os.makedirs(todays_dir, exist_ok=True)

    # Read existing todos
    try:
        with open(f"{todays_dir}/todo.md", "r") as f:
            todos = f.read().splitlines()
    except FileNotFoundError:
        todos = []

    # Process new todo (example: "add: Buy milk")
    instruction = state.get_last_human_message().content
    if instruction.startswith("add:"):
        new_todo = instruction[len("add:"):].strip()
        todos.append(f"- [ ] {new_todo}")
    elif instruction.startswith("complete:"):
        idx = int(instruction.split()[1])-1
        if 0 <= idx < len(todos):
            todos[idx] = todos[idx].replace("[ ]", "[x]")

    # Save updates
    with open(f"{todays_dir}/todo.md", "w") as f:
        f.write("\n".join(todos))

    return [AIMessage(content=f"Updated todo list:\n{''.join(todos)}")]


2. Daily Data Summary Agent


@task
async def data_summary_agent(state: ChatState, files: list):
    todays_dir = f"helper/helper_for_{date.today()}/"

    summary = []
    for file in files:
        if file.startswith("http"):
            # Download URL content
            resp = httpx.get(file)
            content = resp.text[:200] + "..."
        else:
            with open(file, 'r') as f:
                content = f.read()[:200] + "..."

        summary.append(f"---\nSource: {file}\nPreview: {content}")

    with open(f"{todays_dir}/daily_data_summary.md", "a") as f:
        f.write("\n\n".join(summary))

    return [AIMessage(content="Added to daily summary")]


3. Schedule Helper Workflow


import schedule
import time

async def run_helper():
    # Create state for helper workflow
    helper_state = ChatState(...)  # Populate with dummy values

    # Run both agents
    await todo_agent(helper_state)
    await data_summary_agent(helper_state, files=[])

schedule.every().day.at("09:00").do(run_helper)

# Background runner
async def job_runner():
    while True:
        schedule.run_pending()
        await asyncio.sleep(60)



                                                             Phase 3: External API Proxy Agent

                                                      Goal: Securely interface with external LLM APIs

                                                                       Architecture:

 1 Query Preparation Agent
 2 Response Processing Agent

                                                                      Implementation:

- **Query Preparation Agent**
  
  ```python
  @task
  async def prepare_external_query(state: ChatState):
      # Aggregate relevant info
      context = "\n".join([
          msg.content for msg in state.messages
          if isinstance(msg, (HumanMessage, AIMessage))
      ])

      # Redact PII using regex patterns
      sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', context)

      # Format query
      query = f"CONTEXT:\n{sanitized}\nQUESTION:\n{state.last_user_input}"

      return query, state.last_user_input  # Return original for re-insertion
  ```

- **Response Processing Agent**
  
  ```python
  async def process_external_response(original_query, external_response):
      # Re-insert PII using placeholder mapping
      placeholder_map = {
          "[SSN]": re.search(r'\b\d{3}-\d{2}-\d{4}\b', original_query).group()
      }

      processed = external_response
      for placeholder, actual in placeholder_map.items():
          processed = processed.replace(placeholder, actual)

      return processed
  ```



                                                               Phase 4: Integration & Testing

                                                                         Key Areas:

- **Command Testing**
  - Verify `/roll` shows dice results instantly
  - Test `/search [term]` produces proper search results

- **Helper Workflow Verification**
  - Manually trigger helper agents through CLI
  - Check file creation/update in `helper/` directories
  - Confirm scheduled tasks execute at proper intervals

- **API Proxy Validation**
  - Mock external API responses
  - Test PII redaction/re-insertion with sample data
  - Measure latency improvements from parallel processing


                                                                    Risk Mitigation Plan

- **Concurrency Control**
  - Use file locking mechanisms for shared resource access
  - Implement atomic writes for critical files

- **Security Measures**
  - Add encryption for sensitive data in summaries
  - Rate-limit external API calls to prevent abuse

- **Fallback Mechanisms**
  - Cache last-known-good state for helper workflows
  - Implement retry logic for failed API calls


                                                                         Next Steps

- **Implement command handlers first (Phase 1)**
- **Develop helper workflow components (Phase 2)**
- **Create API proxy architecture (Phase 3)**
- **Establish automated testing framework extensions**
- **Perform security audit focusing on PII handling**


