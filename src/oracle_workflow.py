from src.models import ChatState
from langchain_core.messages import BaseMessage, AIMessage
from src.persona_workflows import persona_workflows
from src.agents.persona_classifier_agent import persona_classifier_agent
from src.agents.director_agent import director_agent  # ADD THIS IMPORT
from langchain_core.runnables import RunnableLambda
import logging
import datetime

import chainlit as cl

cl_logger = logging.getLogger("chainlit")


async def oracle_workflow(inputs: dict, state:ChatState, *, config=None) -> list[BaseMessage]:
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

        # Check if persona is unset or force_classify requested
        force_classify = inputs.get("force_classify", False)
        if not getattr(state, "current_persona", None) or force_classify:
            cl_logger.info("Oracle: Running persona classifier...")
            result = await persona_classifier_agent(state, config=config)
            suggested_persona = result.get("persona", "default")
            cl_logger.info(f"Oracle: Classifier suggests persona '{suggested_persona}'")
            state.current_persona = suggested_persona
            # Force update the session AND the state object attribute
            try:
                import chainlit as cl
                cl.user_session.set("current_persona", state.current_persona)
            except Exception:
                pass
            # Defensive: update ChatState attribute explicitly
            if hasattr(state, "__dict__"):
                state.__dict__["current_persona"] = suggested_persona
            append_log(state.current_persona, f"Oracle dispatched to persona workflow: {state.current_persona}")

            # After classifier, update persona in all existing AI message metadata to match new persona
            for msg in state.messages:
                if isinstance(msg, AIMessage):
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["persona"] = state.current_persona

        persona_key = state.current_persona.lower().replace(" ", "_")
        workflow_func = persona_workflows.get(persona_key)

        if not workflow_func:
            cl_logger.warning(
                f"Oracle: Unknown persona '{persona_key}', falling back to default workflow"
            )
            workflow_func = persona_workflows.get("default")

        append_log(state.current_persona, f"Oracle dispatched to persona workflow: {state.current_persona}")

        # Run the director to get list of actions/tools
        try:
            actions = await director_agent(state, config=config)
        except Exception as e:
            cl_logger.error(f"Director agent failed: {e}")
            actions = ["continue_story"]

        from src.agents import agents_map

        # For each tool, run it and append its outputs
        for action in actions:
            tool_func = agents_map.get(action)
            if tool_func:
                try:
                    tool_outputs = await tool_func(state, config=config)
                    # Patch metadata persona to match updated state.current_persona
                    if isinstance(tool_outputs, list):
                        for msg in tool_outputs:
                            if hasattr(msg, "metadata") and isinstance(msg.metadata, dict):
                                msg.metadata.setdefault("persona", state.current_persona)
                        state.messages.extend(tool_outputs)
                    elif isinstance(tool_outputs, dict) and "messages" in tool_outputs:
                        for msg in tool_outputs["messages"]:
                            if hasattr(msg, "metadata") and isinstance(msg.metadata, dict):
                                msg.metadata.setdefault("persona", state.current_persona)
                        state.messages.extend(tool_outputs["messages"])
                except Exception as e:
                    cl_logger.error(f"Tool '{action}' failed: {e}")

        # After all tools, run the persona-specific writer agent
        try:
            response = await workflow_func(inputs, state, config=config)
        except Exception as e:
            cl_logger.error(f"Persona workflow '{persona_key}' failed: {e}")
            response = [
                AIMessage(
                    content="An error occurred in the oracle workflow.",
                    name="error",
                    metadata={"message_id": None},
                )
            ]

        # Defensive: handle dict response (legacy or tool output)
        if isinstance(response, dict) and "messages" in response:
            state.messages.extend(response["messages"])
            return state
        elif isinstance(response, list):
            state.messages.extend(response)
            return state
        elif isinstance(response, ChatState):
            return response
        else:
            return state

    except Exception as e:
        cl_logger.error(f"Oracle workflow failed: {e}")
        from src.storage import append_log
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


async def _oracle_workflow_wrapper(*args, **kwargs):
    config = kwargs.pop("config", None)
    return await oracle_workflow(*args, **kwargs, config=config)


oracle_workflow_runnable = RunnableLambda(_oracle_workflow_wrapper)
