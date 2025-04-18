import os

from browser_use import Agent, Browser, BrowserConfig
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa


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
