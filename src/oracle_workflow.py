from src.models import ChatState
from langchain_core.messages import BaseMessage, AIMessage
from src.persona_workflows import persona_workflows
from src.agents.persona_classifier_agent import persona_classifier_agent
import logging

import chainlit as cl

cl_logger = logging.getLogger("chainlit")


async def oracle_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    try:
        # Defensive: get passed config or empty dict
        if config is None:
            config = {}

        # Support legacy positional args: (messages, previous)
        if not isinstance(inputs, dict):
            # assume *args style call
            args = inputs
            inputs = {}
            if len(args) == 2:
                inputs["messages"] = args[0]
                inputs["previous"] = args[1]
            elif len(args) == 1:
                inputs["messages"] = args[0]
            else:
                # empty or unknown
                pass

        # If called with {"messages": ..., "previous": ...}
        # convert to inputs dict and extract state
        if "previous" in inputs and isinstance(inputs["previous"], ChatState):
            state = inputs["previous"]
            # update state.messages if provided
            if "messages" in inputs and inputs["messages"] is not None:
                state.messages = inputs["messages"]
        elif isinstance(state, ChatState):
            # state is passed explicitly
            pass
        else:
            # fallback: create dummy state
            state = ChatState(messages=inputs.get("messages", []), thread_id="unknown")

        # Check if persona is unset or force_classify requested
        force_classify = inputs.get("force_classify", False)
        if not getattr(state, "current_persona", None) or force_classify:
            cl_logger.info("Oracle: Running persona classifier...")
            result = await persona_classifier_agent(state, config=config)
            suggested_persona = result.get("persona", "default")
            cl_logger.info(f"Oracle: Classifier suggests persona '{suggested_persona}'")
            state.current_persona = suggested_persona

        persona_key = state.current_persona.lower().replace(" ", "_")
        workflow_func = persona_workflows.get(persona_key)

        if not workflow_func:
            cl_logger.warning(
                f"Oracle: Unknown persona '{persona_key}', falling back to default workflow"
            )
            workflow_func = persona_workflows.get("default")

        response = await workflow_func(inputs, state, config=config)

        # Wrap list of messages into ChatState for compatibility
        if isinstance(response, list):
            state.messages.extend(response)
            return state
        elif isinstance(response, ChatState):
            return response
        else:
            # Unexpected return type, wrap into state
            return state

    except Exception as e:
        cl_logger.error(f"Oracle workflow failed: {e}")
        # Wrap error message into ChatState
        error_msg = AIMessage(
            content="An error occurred in the oracle workflow.",
            name="error",
            metadata={"message_id": None},
        )
        state.messages.append(error_msg)
        return state


# Add dummy .ainvoke method so tests patching it don't fail
async def _ainvoke(*args, **kwargs):
    config = kwargs.pop("config", None)
    return await oracle_workflow(*args, **kwargs, config=config)


oracle_workflow.ainvoke = _ainvoke


class OracleWorkflowWrapper:
    async def ainvoke(self, *args, **kwargs):
        config = kwargs.pop("config", None)
        return await oracle_workflow(*args, **kwargs, config=config)

    async def __call__(self, *args, **kwargs):
        # support calling instance directly as async callable
        return await self.ainvoke(*args, **kwargs)

chat_workflow = OracleWorkflowWrapper()
