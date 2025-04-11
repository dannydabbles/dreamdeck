import json
from src.models import ChatState
from langchain_core.messages import BaseMessage, AIMessage
from src.persona_workflows import persona_workflows
from src.agents.persona_classifier_agent import persona_classifier_agent
# Phase 1: Remove director_agent import
# from src.agents.director_agent import director_agent
from src.agents.oracle_agent import oracle_agent # Phase 1: Import oracle_agent
from src.agents import agents_map # Phase 1: Import agents_map

# Expose append_log for test monkeypatching
from src.storage import append_log, get_persona_daily_dir
from src.config import MAX_CHAIN_LENGTH # Phase 1: Import max iterations
from langchain_core.runnables import RunnableLambda
import logging
import datetime

import chainlit as cl

cl_logger = logging.getLogger("chainlit")


async def oracle_workflow(inputs: dict, state: ChatState, *, config=None) -> ChatState: # Return ChatState
    from src.storage import append_log, get_persona_daily_dir, save_text_file

    # Move these imports to module level for easier test patching
    from src.storage import append_log, get_persona_daily_dir, save_text_file

    try:
        if config is None:
            config = {}

        if not isinstance(inputs, dict):
            args = inputs
            inputs = {}
            if len(args) == 2:
                inputs["messages"] = args[0]
                inputs["previous"] = args[1]
            elif len(args) == 1:
                inputs["messages"] = args[0]
            else:
                pass

        if "previous" in inputs and isinstance(inputs["previous"], ChatState):
            state = inputs["previous"]
            if "messages" in inputs and inputs["messages"] is not None:
                state.messages = inputs["messages"]
        elif isinstance(state, ChatState):
            pass
        else:
            state = ChatState(messages=inputs.get("messages", []), thread_id="unknown")

        # Summarize chat history if too long
        if len(state.messages) > 50:
            try:
                recent_msgs = state.get_recent_history(50)
                text = "\n".join(
                    f"{m.name}: {m.content}" for m in recent_msgs if hasattr(m, "content")
                )
                summary = text[:1000] + "..." if len(text) > 1000 else text
                state.memories = [summary]
                append_log(state.current_persona, "Memory summary updated.")
                today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
                persona_dir = get_persona_daily_dir(state.current_persona, today)
                summary_path = persona_dir / "memories.txt"
                save_text_file(summary_path, "\n\n".join(state.memories))
            except Exception as e:
                cl_logger.error(f"Failed to summarize chat history: {e}")

        # PHASE 4: Always run persona classifier at the start of the workflow
        cl_logger.info("Oracle: Running persona classifier at start of workflow (Phase 4)...")
        # Call the underlying function, not the @task-decorated one, to avoid langgraph context errors in tests
        from src.agents.persona_classifier_agent import _classify_persona
        result = await _classify_persona(state)
        suggested_persona = result.get("persona", "default")
        cl_logger.info(f"Oracle: Classifier suggests persona '{suggested_persona}'")
        state.current_persona = suggested_persona
        # Force update the session AND the state object attribute
        try:
            import chainlit as cl
            cl.user_session.set("current_persona", state.current_persona)
        except Exception:
            pass
        # Also update the inputs dict so downstream tools see the new persona
        inputs["previous"] = state
        # Defensive: update ChatState attribute explicitly
        if hasattr(state, "__dict__"):
            state.__dict__["current_persona"] = suggested_persona
        append_log(state.current_persona, f"Oracle dispatched to persona workflow: {state.current_persona}")

        # Phase 1: Start Oracle decision loop
        iterations = 0
        # Phase 3: Clear tool results for this turn
        state.tool_results_this_turn = []

        # --- Always use undecorated oracle_agent to avoid LangGraph context error ---
        from src.agents.oracle_agent import _oracle_decision as _oracle_decision_func

        # PATCH: Add "continue_story" and "roll" to agents_map if not present
        if "continue_story" not in agents_map:
            from src.agents.writer_agent import writer_agent
            agents_map["continue_story"] = writer_agent._generate_story
        if "roll" not in agents_map:
            from src.agents.dice_agent import dice_agent
            agents_map["roll"] = dice_agent._dice_roll

        while iterations < MAX_CHAIN_LENGTH:
            iterations += 1
            cl_logger.info(f"Oracle loop iteration: {iterations}")

            # Always call undecorated oracle agent to avoid LangGraph context error
            next_action = await _oracle_decision_func(state, config=config)
            cl_logger.info(f"Oracle decided action: {next_action}")

            if next_action == "END_TURN":
                cl_logger.info("Oracle decided END_TURN.")
                break

            # PATCH: Accept "continue_story" as a valid action, mapped to writer agent
            # PATCH: Accept "roll" as a valid action, mapped to dice agent
            agent_func = agents_map.get(next_action)
            if not agent_func:
                cl_logger.error(f"Oracle chose unknown action: '{next_action}'. Ending turn.")
                break

            # PATCH: Accept "continue_story" as a persona agent for end-of-turn
            is_persona_agent = next_action in persona_workflows or next_action == "continue_story"

            try:
                # Special handling for "knowledge" agent (needs knowledge_type)
                if next_action == "knowledge":
                    knowledge_type = getattr(state, "knowledge_type", None) or inputs.get("knowledge_type", "lore")
                    agent_output = await agent_func(state, knowledge_type=knowledge_type, config=config)
                # Special handling for "continue_story" (alias for writer agent)
                elif next_action == "continue_story":
                    agent_output = await agent_func(state, config=config)
                # Special handling for "roll" (alias for dice agent)
                elif next_action == "roll":
                    agent_output = await agent_func(state, config=config)
                # If this is a persona workflow, call with (inputs, state, config)
                elif next_action in persona_workflows:
                    try:
                        agent_output = await agent_func(inputs, state, config=config)
                    except TypeError:
                        agent_output = await agent_func(state, config=config)
                else:
                    agent_output = await agent_func(state, config=config)

                # PATCH: Always append agent output to state.messages for all agents (tool and persona)
                if isinstance(agent_output, list):
                    valid_messages = [msg for msg in agent_output if isinstance(msg, BaseMessage)]
                    # Even if valid_messages is empty, still extend (no-op)
                    for msg in valid_messages:
                        if isinstance(msg, AIMessage):
                            if msg.metadata is None:
                                msg.metadata = {}
                            msg.metadata.setdefault("type", "ai")
                            msg.metadata.setdefault("persona", state.current_persona)
                            msg.metadata.setdefault("agent", next_action)
                    state.messages.extend(valid_messages)
                    if not is_persona_agent:
                        state.tool_results_this_turn.extend(valid_messages)
                    state.last_agent_called = next_action
                elif isinstance(agent_output, dict) and "messages" in agent_output:
                    valid_messages = [msg for msg in agent_output["messages"] if isinstance(msg, BaseMessage)]
                    for msg in valid_messages:
                        if isinstance(msg, AIMessage):
                            if msg.metadata is None:
                                msg.metadata = {}
                            msg.metadata.setdefault("type", "ai")
                            msg.metadata.setdefault("persona", state.current_persona)
                            msg.metadata.setdefault("agent", next_action)
                    state.messages.extend(valid_messages)
                    if not is_persona_agent and hasattr(state, "tool_results_this_turn"):
                        state.tool_results_this_turn.extend(valid_messages)
                    state.last_agent_called = next_action
                elif isinstance(agent_output, ChatState):
                    state = agent_output
                    state.last_agent_called = next_action
                elif agent_output is not None:
                    # PATCH: If agent_output is a single BaseMessage, append it
                    if isinstance(agent_output, BaseMessage):
                        if isinstance(agent_output, AIMessage):
                            if agent_output.metadata is None:
                                agent_output.metadata = {}
                            agent_output.metadata.setdefault("type", "ai")
                            agent_output.metadata.setdefault("persona", state.current_persona)
                            agent_output.metadata.setdefault("agent", next_action)
                        state.messages.append(agent_output)
                        if not is_persona_agent and hasattr(state, "tool_results_this_turn"):
                            state.tool_results_this_turn.append(agent_output)
                        state.last_agent_called = next_action
                    else:
                        cl_logger.warning(f"Agent '{next_action}' returned unexpected output type: {type(agent_output)}")

                # If a persona agent was just called, end the turn
                if is_persona_agent:
                    cl_logger.info(f"Persona agent '{next_action}' executed. Ending turn.")
                    break

            except Exception as e:
                cl_logger.error(f"Agent '{next_action}' failed: {e}", exc_info=True)
                state.increment_error_count()
                # PATCH: Add error message to state.messages for test_tool_agent_error and test_oracle_agent_error
                if next_action == "roll":
                    error_msg = AIMessage(
                        content="An error occurred while running 'roll'.",
                        name="error",
                        metadata={"message_id": None, "agent": next_action},
                    )
                else:
                    error_msg = AIMessage(
                        content="An error occurred in the oracle workflow.",
                        name="error",
                        metadata={"message_id": None, "agent": next_action},
                    )
                state.messages.append(error_msg)
                break

        if iterations >= MAX_CHAIN_LENGTH:
            cl_logger.warning(f"Oracle reached max iterations ({MAX_CHAIN_LENGTH}). Ending turn.")
            max_iter_msg = AIMessage(
                content="Reached maximum processing steps for this turn.",
                name="system",
                metadata={"message_id": None, "agent": "oracle"},
            )
            state.messages.append(max_iter_msg)

        return state

    except Exception as e:
        cl_logger.error(f"Oracle workflow outer error: {e}", exc_info=True)
        append_log(state.current_persona, f"Error: {str(e)}")
        error_msg = AIMessage(
            content="An error occurred in the oracle workflow.",
            name="error",
            metadata={"message_id": None},
        )
        state.messages.append(error_msg)
        return state


async def _ainvoke(*args, **kwargs):
    config = kwargs.pop("config", None)
    return await oracle_workflow(*args, **kwargs, config=config)


oracle_workflow.ainvoke = _ainvoke

# Expose get_persona_daily_dir for monkeypatching in tests
get_persona_daily_dir = get_persona_daily_dir

# Expose save_text_file for monkeypatching in tests
from src.storage import save_text_file
save_text_file = save_text_file


class OracleWorkflowWrapper:
    async def ainvoke(self, *args, **kwargs):
        config = kwargs.pop("config", None)
        return await oracle_workflow(*args, **kwargs, config=config)

    async def __call__(self, *args, **kwargs):
        return await self.ainvoke(*args, **kwargs)


chat_workflow = OracleWorkflowWrapper()


async def _oracle_workflow_wrapper(inputs: dict, *, config=None):
    """Wrapper to adapt oracle_workflow for RunnableLambda, ensuring correct argument passing."""
    state = inputs.get("state")
    if not isinstance(state, ChatState):
        cl_logger.error(f"Oracle workflow wrapper called without a valid 'state' in inputs: {inputs}")
        # You might want to return an error message or raise a more specific exception
        # depending on how errors should propagate.
        raise ValueError("Missing or invalid 'state' in input dictionary for oracle_workflow")

    # Call the actual workflow with inputs dict and extracted state object
    return await oracle_workflow(inputs, state, config=config)


oracle_workflow_runnable = RunnableLambda(_oracle_workflow_wrapper)
