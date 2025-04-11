from src.config import config, cl_logger
import os
import datetime
import zoneinfo
import re
from chainlit import Message as CLMessage
import chainlit as cl
from langgraph.func import task
from langchain_core.messages import AIMessage
from src.models import ChatState
from langchain_openai import ChatOpenAI
from jinja2 import Template


@cl.step(name="Report Agent", type="tool")
async def _generate_report(state: ChatState, **kwargs) -> list[AIMessage]:
    """
    Generates a daily report summarizing TODOs and listing image filenames.

    Note:
    - The LLM **does NOT receive** any image data, URLs, or embeddings.
    - Only a plain text list of image filenames (like 'map.png') is included in the prompt.
    - The LLM cannot analyze or "see" the images, just their names.
    """
    try:
        pacific = zoneinfo.ZoneInfo("America/Los_Angeles")
        current_date = datetime.datetime.now(pacific).strftime("%Y-%m-%d")
        base_dir = config.todo_dir_path

        all_todos = []
        all_images = []

        if os.path.exists(base_dir):
            for persona_name in os.listdir(base_dir):
                persona_dir = os.path.join(base_dir, persona_name, current_date)
                todo_file = os.path.join(persona_dir, config.todo_file_name)
                if os.path.exists(todo_file):
                    try:
                        with open(todo_file, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                all_todos.append(
                                    f"### Persona: {persona_name}\n{content}"
                                )
                    except Exception as e:
                        cl_logger.warning(f"Failed to read {todo_file}: {e}")

                # Collect image files in the same directory
                if os.path.exists(persona_dir):
                    for fname in os.listdir(persona_dir):
                        if fname.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".gif", ".webp")
                        ):
                            image_path = os.path.join(persona_dir, fname)
                            all_images.append((persona_name, image_path))

        todo_content = (
            "\n\n".join(all_todos) if all_todos else "No TODOs found for any persona."
        )

        prompt_template_str = config.loaded_prompts.get(
            "daily_report_prompt", ""
        ).strip()

        # Build plain text list of image filenames (not URLs or data)
        if all_images:
            image_list_str = "\n".join(
                f"- {persona}: {os.path.basename(path)}" for persona, path in all_images
            )
        else:
            image_list_str = "No images found for today."

        if not prompt_template_str:
            cl_logger.error("Daily report prompt template is empty!")
            prompt = f"TODO list:\n{todo_content}\n\nImages:\n{image_list_str}"
        else:
            template = Template(prompt_template_str)
            prompt = template.render(todo_list=todo_content, image_list=image_list_str)

        cl_logger.info(f"Daily report prompt:\n{prompt}")

        user_settings = cl.user_session.get("chat_settings", {})
        final_temp = user_settings.get("report_temp", 0.3)
        final_endpoint = user_settings.get("report_endpoint") or config.openai.get(
            "base_url"
        )
        final_max_tokens = user_settings.get("report_max_tokens", 500)

        llm = ChatOpenAI(
            base_url=final_endpoint,
            temperature=final_temp,
            max_tokens=final_max_tokens,
            streaming=False,
            verbose=True,
            timeout=config.llm.timeout,
        )

        response = await llm.ainvoke([("system", prompt)])
        report_text = response.content.strip()

        cl_msg = CLMessage(
            content=f"📋 Daily Report:\n{report_text}",
            parent_id=None,
        )
        await cl_msg.send()

        return [
            AIMessage(
                content=report_text,
                name="report",
                metadata={"message_id": cl_msg.id},
            )
        ]

    except Exception as e:
        cl_logger.error(f"Report agent failed: {e}")
        return [
            AIMessage(
                content="Report generation failed.",
                name="error",
                metadata={"message_id": None},
            )
        ]


# Refactored: report_agent is now a stateless, LLM-backed function (task)
@task
async def report_agent(state: ChatState, **kwargs) -> list[AIMessage]:
    return await _generate_report(state, **kwargs)
