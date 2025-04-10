import click
import asyncio
import json
from pathlib import Path
from src.models import ChatState, HumanMessage
from src.oracle_workflow import oracle_workflow

STATE_FILE = Path("chat_state.json")


def save_state(state: ChatState, path: Path = STATE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, indent=2)


def load_state(path: Path = STATE_FILE) -> ChatState:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ChatState.model_validate(data)
    else:
        return ChatState(messages=[], thread_id="cli_thread")


@click.group()
def cli():
    """Dreamdeck CLI"""
    pass


@cli.command()
@click.option("--persona", default=None, help="Persona to use (optional)")
@click.option("--input", "user_input", required=True, help="User input message")
def chat(persona, user_input):
    """Send a message to the AI and get a response."""
    state = load_state()

    # Set persona if provided
    if persona:
        state.current_persona = persona

    # Add user message
    user_msg = HumanMessage(content=user_input, name="Player")
    state.messages.append(user_msg)

    # Run oracle_workflow
    async def run():
        updated_state = await oracle_workflow({"messages": state.messages, "previous": state}, state)
        # Save updated state
        save_state(updated_state)
        # Print AI responses
        for msg in updated_state.messages[len(state.messages):]:
            if hasattr(msg, "content"):
                print(f"{msg.name or 'AI'}: {msg.content}")

    asyncio.run(run())


@cli.command("list-personas")
def list_personas():
    """List available personas."""
    from src.agents.persona_classifier_agent import PERSONA_LIST
    for persona in PERSONA_LIST:
        print(persona)


@cli.command("switch-persona")
@click.argument("persona_name")
def switch_persona(persona_name):
    """Switch current persona."""
    state = load_state()
    state.current_persona = persona_name
    save_state(state)
    print(f"Persona switched to: {persona_name}")


@cli.command()
@click.option("--format", type=click.Choice(["json", "markdown"]), default="markdown")
@click.option("--output", type=click.Path(), default="chat_export.md")
def export(format, output):
    """Export chat history."""
    state = load_state()
    if format == "json":
        with open(output, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(), f, indent=2)
    else:
        lines = []
        for msg in state.messages:
            if msg.__class__.__name__ == "HumanMessage":
                lines.append(f"**Player:** {msg.content}")
            else:
                lines.append(f"**{msg.name or 'AI'}:** {msg.content}")
        with open(output, "w", encoding="utf-8") as f:
            f.write("\n\n".join(lines))
    print(f"Chat exported to {output}")


if __name__ == "__main__":
    cli()
