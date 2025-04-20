import asyncio
import json
import logging
import os
import subprocess
import sys
import warnings
from typing import Any

logging.basicConfig()

from browser_use import Agent, Browser, BrowserConfig
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_google_genai import ChatGoogleGenerativeAI
from preprocessor import PreprocessLLM

warnings.filterwarnings("ignore")

logger = logging.getLogger("silanthro/basic-browser-use")


def ensure_playwright_chromium():
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    logger.info("Chromium is ready.")


# TODO: Consider setting up an init script for stores
ensure_playwright_chromium()


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
    metadata_queue = asyncio.Queue()

    async def metadata_stream():
        while True:
            metadata = await metadata_queue.get()
            if metadata is None:
                break
            yield metadata

    class ContextCallbackHandler(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs: Any) -> None:
            additional_kwargs = response.generations[0][0].message.additional_kwargs
            if additional_kwargs:
                metadata_str = additional_kwargs.get("function_call", {}).get(
                    "arguments", "{}"
                )
                metadata = json.loads(metadata_str)
                safe_create_task(metadata_queue.put(metadata))

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
        await metadata_queue.put(None)  # stop signal

    asyncio.create_task(run_agent())

    async for metadata in metadata_stream():
        yield {"type": "metadata", "data": metadata}

    final_result = await final_result_future
    yield {"type": "result", "data": final_result}
