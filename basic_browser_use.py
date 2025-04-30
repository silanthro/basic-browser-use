import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import warnings
from typing import Any

logging.basicConfig()

from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
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


def get_context_callback_handler(metadata_queue):
    class ContextCallbackHandler(BaseCallbackHandler):
        _actions = []
        _action_results = []

        def on_llm_start(
            self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
        ) -> None:
            content = prompts[0]
            actions_rgx = re.compile(
                "\\[Your task history memory starts here\\](?P<actions>.*?)\\[Task history memory ends\\]",
                re.DOTALL,
            )
            action_result_rgx = re.compile(
                "Action result:(?P<result>.*?)\n(AI|Human):", re.DOTALL
            )
            current_url_rgx = re.compile(
                "Current url: (?P<url>.*?)\nAvailable tabs:", re.DOTALL
            )

            # Get current URL
            url_match = current_url_rgx.search(content)
            url = None
            if url_match:
                url = url_match.groupdict().get("url")

            # Get action results if any
            match = actions_rgx.search(content)
            latest_results = []
            if match:
                raw_actions = match.groupdict().get("actions")
                matches = action_result_rgx.findall(raw_actions)
                for match in matches:
                    latest_results.append(match[0].strip())
            if len(latest_results) > len(self._action_results):
                self._action_results.append(latest_results[-1])
                # Queue action with corresponding result
                safe_create_task(
                    metadata_queue.put(
                        {
                            "action": self._actions[-1],
                            "result": latest_results[-1],
                            "url": url,
                        }
                    )
                )

        def on_llm_end(self, response, **kwargs: Any) -> None:
            additional_kwargs = response.generations[0][0].message.additional_kwargs
            if additional_kwargs:
                metadata_str = additional_kwargs.get("function_call", {}).get(
                    "arguments", "{}"
                )
                metadata = json.loads(metadata_str)
                action = metadata.get("action")
                if action:
                    # Action will be {<action_name_str>: <args dict>}
                    self._actions.append(action[0])
                safe_create_task(metadata_queue.put(metadata))

    return ContextCallbackHandler()


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

    async def run_agent():
        callback_handler = get_context_callback_handler(metadata_queue)
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

    while True:
        metadata = await metadata_queue.get()
        if metadata is None:
            break
        yield {"type": "metadata", "data": metadata}

    final_result = await final_result_future
    yield {"type": "result", "data": final_result}


async def stream_browser_agent_gui(task: str):
    """
    Do research on the Internet via a browser-based agent with a GUI-based browser
    Some websites detect and block headless traffic, which can be mitigated with a GUI-based browser
    Because this opens a GUI browser, this should be used only if stream_browser_agent is unsuccessful
    and always ask for user permission to open a GUI browser before running this
    Ignore this if run_browser_agent is available

    Args:
    - task (str): Task for the agent to fulfill

    Returns:
        A generator that yields messages corresponding to the progress on the task
    """

    final_result_future = asyncio.Future()
    metadata_queue = asyncio.Queue()

    async def run_agent():
        callback_handler = get_context_callback_handler(metadata_queue)
        callback_manager = CallbackManager([callback_handler])

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            api_key=os.getenv("GEMINI_API_KEY"),
            callback_manager=callback_manager,
        )

        config = BrowserConfig(headless=False, disable_security=False)
        browser = Browser(config)
        context_config = BrowserContextConfig(
            browser_window_size={"width": 640, "height": 480},
            highlight_elements=False,
        )
        context = BrowserContext(browser=browser, config=context_config)
        agent = Agent(
            task=task,
            llm=PreprocessLLM(llm),
            use_vision=False,
            browser_context=context,
            browser=browser,
        )

        history = await agent.run()
        final_result_future.set_result(history.final_result())
        await browser.close()
        await metadata_queue.put(None)  # stop signal

    asyncio.create_task(run_agent())

    while True:
        metadata = await metadata_queue.get()
        if metadata is None:
            break
        yield {"type": "metadata", "data": metadata}

    final_result = await final_result_future
    yield {"type": "result", "data": final_result}
