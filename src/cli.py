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

def run_async(coro):
    """
    Run an async coroutine in a way that works in both normal and test (event loop) environments.
    Uses a helper to avoid conflicts with test monkeypatching.
    """
    # Helper for test compatibility: if DREAMDECK_TEST_MODE is set, always use run_until_complete
    import os
    if os.environ.get("DREAMDECK_TEST_MODE") == "1":
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
    return asyncio.run(coro)

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
            result = run_async(agent(state))
        except RuntimeError as e:
            print(f"Error running agent: {e}")
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
            result = run_async(supervisor(state))
        except RuntimeError as e:
            print(f"Error running workflow: {e}")
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
        # Import save_state here so it can be patched in tests
        from src.storage import save_state
        save_state(state, args.output)
        print(f"State exported to {args.output}")
        return

    parser.print_help()

if __name__ == "__main__":
    main()
