import re
from typing import Any, List, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.runnables.base import RunnableBinding
from pydantic import PrivateAttr


class PreprocessLLM(BaseChatModel):
    _base_llm: BaseChatModel = PrivateAttr()
    _system_prefix: str = PrivateAttr(default="")

    def __init__(self, base_llm: BaseChatModel, system_prefix: str = ""):
        super().__init__()
        self._base_llm = base_llm
        self._system_prefix = system_prefix

    def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        return self._base_llm._generate(self._prepare(messages), **kwargs)

    async def _agenerate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        return await self._base_llm._agenerate(self._prepare(messages), **kwargs)

    def _prepare(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        modified_messages = []
        # Remove base64 content
        pattern = r"data:[^;]+;base64,[a-zA-Z0-9+/=\n\r]+"
        for msg in messages:
            cleaned = re.sub(pattern, "[base64 removed]", msg.content)
            msg.content = cleaned
            modified_messages.append(msg)

        return modified_messages

    @property
    def _llm_type(self) -> str:
        return "preprocess-wrapper"

    def with_structured_output(
        self,
        schema: Optional[Type[Any]] = None,
        *,
        include_raw: bool = False,
        method: Optional[str] = "function_calling",
    ) -> Runnable:
        structured_llm = self._base_llm.with_structured_output(
            schema,
            include_raw=include_raw,
            method=method,
        )

        # Return a bound runnable that preprocesses input messages
        class PreprocessedLLM(RunnableBinding):
            def __init__(self, runnable):
                super().__init__(bound=runnable)

            async def ainvoke(self, input: Any, **kwargs) -> Any:
                input = self._clean_input(input)
                return await self.bound.ainvoke(input, **kwargs)

            def invoke(self, input: Any, **kwargs) -> Any:
                input = self._clean_input(input)
                return self.bound.invoke(input, **kwargs)

            def _clean_input(self, input: Any) -> Any:
                if isinstance(input, list) and all(
                    isinstance(m, BaseMessage) for m in input
                ):
                    return self_outer._prepare(input)
                return input

        self_outer = self  # To reference self in nested class
        return PreprocessedLLM(structured_llm)
