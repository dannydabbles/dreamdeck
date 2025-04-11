import json
from src.models import ChatState
from langchain_core.messages import BaseMessage, AIMessage
from src.persona_workflows import persona_workflows
from src.agents.persona_classifier_agent import persona_classifier_agent
# Phase 1: Remove director_agent import
# from src.agents.director_agent import director_agent
from src.agents.oracle_agent import oracle_agent # Phase 1: Import oracle_agent
from src.agents import agents_map # Phase 1: Import agents_map
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

        # --- PATCH: Use undecorated oracle_agent in tests to avoid LangGraph context error ---
        from src.agents.oracle_agent import _oracle_decision as _oracle_decision_func
        use_undecorated_oracle = False
        import os
        if os.environ.get("DREAMDECK_TEST_MODE") == "1" or getattr(oracle_agent, "_force_undecorated", False):
            use_undecorated_oracle = True

        while iterations < MAX_CHAIN_LENGTH:
            iterations += 1
            cl_logger.info(f"Oracle loop iteration: {iterations}")

            # Call Oracle agent to decide next action
            if use_undecorated_oracle:
                next_action = await _oracle_decision_func(state, config=config)
            else:
                next_action = await oracle_agent(state, config=config)
            cl_logger.info(f"Oracle decided action: {next_action}")

            if next_action == "END_TURN":
                cl_logger.info("Oracle decided END_TURN.")
                break

            # Get the agent function from the map
            agent_func = agents_map.get(next_action)

            if not agent_func:
                cl_logger.error(f"Oracle chose unknown action: '{next_action}'. Ending turn.")
                break

            # Check if the chosen action is a persona agent (signaling end of tool chain)
            is_persona_agent = next_action in persona_workflows

            try:
                # Execute the chosen agent/tool
                # If this is a persona workflow, call with (inputs, state, config)
                if next_action in persona_workflows:
                    try:
                        agent_output = await agent_func(inputs, state, config=config)
                    except TypeError:
                        # For test monkeypatching: allow fallback to (state, **kwargs)
                        agent_output = await agent_func(state, config=config)
                else:
                    agent_output = await agent_func(state, config=config)

                # Process output: Append to messages and potentially tool_results_this_turn
                if isinstance(agent_output, list):
                    valid_messages = [msg for msg in agent_output if isinstance(msg, BaseMessage)]
                    if valid_messages:
                        # Patch metadata for consistency (Phase 2 refinement needed)
                        for msg in valid_messages:
                            if isinstance(msg, AIMessage):
                                if msg.metadata is None:
                                    msg.metadata = {}
                                msg.metadata.setdefault("type", "ai")
                                msg.metadata.setdefault("persona", state.current_persona)
                                msg.metadata.setdefault("agent", next_action) # Track which agent generated it
                        state.messages.extend(valid_messages)
                        # Phase 3: Add tool results if not a persona agent
                        if not is_persona_agent:
                            state.tool_results_this_turn.extend(valid_messages)
                        state.last_agent_called = next_action # Track successful agent call
                elif isinstance(agent_output, dict) and "messages" in agent_output:
                     # Handle dict output (legacy or specific tools)
                     valid_messages = [msg for msg in agent_output["messages"] if isinstance(msg, BaseMessage)]
                     if valid_messages:
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
                    # If agent returns full state, update state (less common)
                    state = agent_output
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
                # Add error message to state? Or just log? For now, log and break.
                error_msg = AIMessage(
                    content=f"An error occurred while running '{next_action}'.",
                    name="error",
                    metadata={"message_id": None, "agent": next_action},
                )
                state.messages.append(error_msg)
                break # Stop processing on agent error

        if iterations >= MAX_CHAIN_LENGTH:
            cl_logger.warning(f"Oracle reached max iterations ({MAX_CHAIN_LENGTH}). Ending turn.")
            # Optionally add a message indicating max iterations reached
            max_iter_msg = AIMessage(
                content="Reached maximum processing steps for this turn.",
                name="system",
                metadata={"message_id": None, "agent": "oracle"},
            )
            state.messages.append(max_iter_msg)

        # Return the final state after the loop finishes or breaks
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
