from src.config import config, cl_logger
import os
import datetime
import re
from chainlit import Message as CLMessage
import chainlit as cl
from langgraph.func import task
from langchain_core.messages import AIMessage
from src.models import ChatState
from langchain_openai import ChatOpenAI
from jinja2 import Template

@cl.step(name="Report Agent", type="tool")
async def _generate_report(state: ChatState) -> list[AIMessage]:
    try:
        persona = getattr(state, "current_persona", "Default")
        current_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        persona_safe = re.sub(r'[^\w\-_. ]', '_', persona)
        dir_path = os.path.join(config.todo_dir_path, persona_safe, current_date)
        file_path = os.path.join(dir_path, config.todo_file_name)

        todo_content = ""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                todo_content = f.read()

        prompt_template_str = config.loaded_prompts.get("daily_report_prompt", "").strip()
        if not prompt_template_str:
            cl_logger.error("Daily report prompt template is empty!")
            prompt = f"TODO list:\n{todo_content}"
        else:
            template = Template(prompt_template_str)
            prompt = template.render(todo_list=todo_content)

        cl_logger.info(f"Daily report prompt:\n{prompt}")

        user_settings = cl.user_session.get("chat_settings", {})
        final_temp = user_settings.get("report_temp", 0.3)
        final_endpoint = user_settings.get("report_endpoint") or config.openai.get("base_url")
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
        return [AIMessage(content="Report generation failed.", name="error", metadata={"message_id": None})]

@task
async def report_agent(state: ChatState) -> list[AIMessage]:
    return await _generate_report(state)
