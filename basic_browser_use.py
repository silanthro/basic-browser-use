import asyncio
import json
import logging
import os
import warnings
from typing import Any

logging.basicConfig()

from browser_use import Agent, Browser, BrowserConfig
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_google_genai import ChatGoogleGenerativeAI
from preprocessor import PreprocessLLM

warnings.filterwarnings("ignore")


async def run_browser_agent(task: str) -> str:
    """
    Do research on the Internet via a browser-based agent

    Args:
    - task (str): Task for the agent to fulfill

    Returns:
        A string representing the result of the task
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    config = BrowserConfig(headless=True, disable_security=False)
    browser = Browser(config)
    agent = Agent(
        task=task,
        llm=llm,
        use_vision=False,
        browser=browser,
    )
    history = await agent.run()
    result = history.final_result()
    return result


def safe_create_task(coro):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)


async def stream_browser_agent(task: str):
    """
    Do research on the Internet via a browser-based agent
    Ignore this if run_browser_agent is available

    Args:
    - task (str): Task for the agent to fulfill

    Returns:
        A generator that yields messages corresponding to the progress on the task
    """

    final_result_future = asyncio.Future()
    memory_queue = asyncio.Queue()

    async def memory_stream():
        while True:
            memory = await memory_queue.get()
            if memory is None:
                break
            yield memory

    class ContextCallbackHandler(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs: Any) -> None:
            additional_kwargs = response.generations[0][0].message.additional_kwargs
            if additional_kwargs:
                metadata_str = additional_kwargs.get("function_call", {}).get(
                    "arguments", "{}"
                )
                metadata = json.loads(metadata_str)
                # I want to yield this
                memory = metadata.get("current_state", {}).get("memory")
                safe_create_task(memory_queue.put(memory))

    async def run_agent():
        callback_handler = ContextCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            api_key=os.getenv("GEMINI_API_KEY"),
            callback_manager=callback_manager,
        )

        config = BrowserConfig(headless=True, disable_security=False)
        browser = Browser(config)
        agent = Agent(
            task=task,
            llm=PreprocessLLM(llm),
            use_vision=False,
            browser=browser,
        )

        history = await agent.run()
        final_result_future.set_result(history.final_result())
        await memory_queue.put(None)  # stop signal

    asyncio.create_task(run_agent())

    async for memory in memory_stream():
        yield {"type": "memory", "data": memory}

    final_result = await final_result_future
    yield {"type": "result", "data": final_result}
