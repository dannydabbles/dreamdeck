"""
Dreamdeck CLI: Command-line access to agents, tools, and workflows.

Usage examples:
    python -m src.cli list-agents
    python -m src.cli run-agent writer --persona "Therapist" --input "My character is sad"
    python -m src.cli run-workflow supervisor --input "Roll for initiative"
    python -m src.cli export-state --thread-id t1 --output state.json
"""

import argparse
import asyncio
import sys
from src.models import ChatState
from src.agents.registry import get_agent, list_agents
from src.supervisor import supervisor

# Import save_state for patching in tests
from src.storage import save_state

def main():
    parser = argparse.ArgumentParser(description="Dreamdeck CLI")
    subparsers = parser.add_subparsers(dest="command")

    # List agents/tools
    subparsers.add_parser("list-agents", help="List all available agents/tools")

    # Run agent
    run_agent_parser = subparsers.add_parser("run-agent", help="Run a specific agent")
    run_agent_parser.add_argument("agent", help="Agent name")
    run_agent_parser.add_argument("--persona", help="Persona name", default="Default")
    run_agent_parser.add_argument("--input", help="Input text", required=True)

    # Run workflow (supervisor)
    run_workflow_parser = subparsers.add_parser("run-workflow", help="Run the supervisor workflow")
    run_workflow_parser.add_argument("--persona", help="Persona name", default="Default")
    run_workflow_parser.add_argument("--input", help="Input text", required=True)

    # Export state
    export_parser = subparsers.add_parser("export-state", help="Export state to JSON")
    export_parser.add_argument("--thread-id", help="Thread ID", required=True)
    export_parser.add_argument("--output", help="Output file", required=True)

    args = parser.parse_args()

    if args.command == "list-agents":
        print("Available agents/tools:")
        for name, desc in list_agents():
            print(f"  {name:20} {desc}")
        return

    if args.command == "run-agent":
        agent = get_agent(args.agent)
        if not agent:
            print(f"Unknown agent: {args.agent}")
            sys.exit(1)
        state = ChatState(messages=[], thread_id="cli", current_persona=args.persona)
        from langchain_core.messages import HumanMessage
        state.messages.append(HumanMessage(content=args.input, name="Player"))
        try:
            # Use asyncio.run only if not already in an event loop (for test compatibility)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # If already in an event loop (e.g., pytest-asyncio), run as a task
                import nest_asyncio
                nest_asyncio.apply()
                result = loop.run_until_complete(agent(state))
            else:
                result = asyncio.run(agent(state))
        except RuntimeError as e:
            if "asyncio.run()" in str(e):
                print(f"Error running agent: {e}")
                raise
            raise
        except Exception as e:
            print(f"Error running agent: {e}")
            raise
        print("Agent output:")
        for msg in result:
            print(f"{msg.name}: {msg.content}")
        return

    if args.command == "run-workflow":
        state = ChatState(messages=[], thread_id="cli", current_persona=args.persona)
        from langchain_core.messages import HumanMessage
        state.messages.append(HumanMessage(content=args.input, name="Player"))
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                result = loop.run_until_complete(supervisor(state))
            else:
                result = asyncio.run(supervisor(state))
        except RuntimeError as e:
            if "asyncio.run()" in str(e):
                print(f"Error running workflow: {e}")
                raise
            raise
        except Exception as e:
            print(f"Error running workflow: {e}")
            raise
        print("Workflow output:")
        for msg in result:
            print(f"{msg.name}: {msg.content}")
        return

    if args.command == "export-state":
        state = ChatState(messages=[], thread_id=args.thread_id)
        # Call save_state in a way that can be patched/mocked in tests
        save_state(state, args.output)
        print(f"State exported to {args.output}")
        return

    parser.print_help()

if __name__ == "__main__":
    main()
