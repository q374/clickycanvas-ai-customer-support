"""受限客服工具执行器和可审计的工具调用轨迹。"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any, Callable


class ToolBudgetExceeded(RuntimeError):
    """单轮工具调用超过安全上限。"""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    status: str = "completed"


class BoundedToolExecutor:
    def __init__(self, max_calls: int = 3) -> None:
        self.max_calls = max_calls
        self.calls: list[ToolCall] = []

    def invoke(self, name: str, arguments: dict[str, Any], handler: Callable[[], Any]) -> Any:
        if len(self.calls) >= self.max_calls:
            self.calls.append(ToolCall(name=name, arguments=_safe_arguments(arguments), status="blocked_budget"))
            raise ToolBudgetExceeded(f"tool call budget exceeded: {self.max_calls}")
        safe_arguments = _safe_arguments(arguments)
        try:
            result = handler()
        except Exception:
            self.calls.append(ToolCall(name=name, arguments=safe_arguments, status="failed"))
            raise
        self.calls.append(ToolCall(name=name, arguments=safe_arguments))
        return result

    def trace(self) -> dict[str, Any]:
        return {
            "mode": "bounded_tool_agent",
            "max_tool_calls": self.max_calls,
            "tool_calls": [
                {"name": call.name, "arguments": call.arguments, "status": call.status}
                for call in self.calls
            ],
            "verified": all(call.status == "completed" for call in self.calls),
        }


ACTIVE_TOOL_EXECUTOR: contextvars.ContextVar[BoundedToolExecutor | None] = contextvars.ContextVar(
    "active_tool_executor", default=None
)


def invoke_tool(name: str, arguments: dict[str, Any], handler: Callable[[], Any]) -> Any:
    executor = ACTIVE_TOOL_EXECUTOR.get()
    return executor.invoke(name, arguments, handler) if executor is not None else handler()


def _safe_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    safe = dict(arguments)
    for key in ("demo_access_code", "access_code"):
        if key in safe:
            safe[key] = "[redacted]"
    return safe
