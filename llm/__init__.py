"""LLM 后端抽象层。

M2 只有 API 后端与测试用的脚本后端；M3 在此新增 GLM-4V、Llama 3.2
Vision 与本地微调模型的实现，Agent 层与 Loop 层不动。

后端实现**延迟导入**：`QwenVLAPIBackend` 会拉起 langchain 与 openai，
只想用 `ScriptedBackend` 跑单元测试时不该付这个代价。
"""

from llm.base import (
    ActionIntent,
    CostInfo,
    HistoryStep,
    LLMBackend,
    LLMBackendError,
    PriceSheet,
    TokenUsage,
)
from llm.fake import ScriptedBackend

__all__ = [
    "ActionIntent",
    "CostInfo",
    "HistoryStep",
    "LLMBackend",
    "LLMBackendError",
    "PriceSheet",
    "ScriptedBackend",
    "TokenUsage",
]
