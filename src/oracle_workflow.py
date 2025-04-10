from src.models import ChatState
from langchain_core.messages import BaseMessage, AIMessage
from src.persona_workflows import persona_workflows
from src.agents.persona_classifier_agent import persona_classifier_agent
import logging

cl_logger = logging.getLogger("chainlit")


async def oracle_workflow(inputs: dict, state: ChatState) -> list[BaseMessage]:
    try:
        # Check if persona is unset or force_classify requested
        force_classify = inputs.get("force_classify", False)
        if not getattr(state, "current_persona", None) or force_classify:
            cl_logger.info("Oracle: Running persona classifier...")
            result = await persona_classifier_agent(state)
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

        response = await workflow_func(inputs, state)
        return response

    except Exception as e:
        cl_logger.error(f"Oracle workflow failed: {e}")
        return [
            AIMessage(
                content="An error occurred in the oracle workflow.",
                name="error",
                metadata={"message_id": None},
            )
        ]
