import os
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.callbacks import BaseCallbackHandler, StdOutCallbackHandler
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()


class TelemetryMonitorHandler(BaseCallbackHandler):
    """按 run_id 隔离的结构化性能与异常监控处理器。"""

    def __init__(self, trace_name: str = "langchain_run") -> None:
        super().__init__()
        self.trace_name = trace_name
        self._start_times: Dict[UUID, float] = {}
        self.records: List[Dict[str, Any]] = []

    def _record(self, event: str, run_id: UUID, extra: Optional[Dict[str, Any]] = None) -> None:
        try:
            start = self._start_times.get(run_id)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2) if start is not None else None
            if event.endswith("_end") or event.endswith("_error"):
                self._start_times.pop(run_id, None)

            payload: Dict[str, Any] = {
                "trace": self.trace_name,
                "event": event,
                "run_id": str(run_id),
                "elapsed_ms": elapsed_ms,
            }
            if extra:
                payload.update(extra)
            self.records.append(payload)
        except Exception as cb_err:
            # 监控自身异常不应影响主业务链路
            print(f"[TelemetryMonitorHandler Warning] {cb_err}")

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.perf_counter()
        name = (serialized or {}).get("name") or kwargs.get("name") or "Chain"
        self._record(
            "chain_start",
            run_id,
            {"component": name, "parent_run_id": str(parent_run_id) if parent_run_id else None},
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._record("chain_end", run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._record("chain_error", run_id, {"error": repr(error)})

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.perf_counter()
        model_name = (
            (kwargs.get("invocation_params") or {}).get("model")
            or (serialized or {}).get("name")
            or "ChatModel"
        )
        self._record("chat_model_start", run_id, {"model": model_name})

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._record("chat_model_end", run_id)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._record("chat_model_error", run_id, {"error": repr(error)})

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.perf_counter()
        tool_name = (serialized or {}).get("name") or kwargs.get("name") or "Tool"
        self._record("tool_start", run_id, {"tool": tool_name, "input": input_str})

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._record("tool_end", run_id, {"output_preview": str(output)[:120]})

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._record("tool_error", run_id, {"error": repr(error)})


@tool
def fetch_financial_snapshot(query: str) -> str:
    """查询最新财务数据快照（此处为示例占位，生产环境请替换为真实财报/行情/内部财务 API）。"""
    return (
        f"[示例数据源返回] 针对查询「{query}」：尚未配置外部实时财报接口；"
        "请接入企业财务数据库或合规金融数据接口以获取最新经审计/披露数据。"
    )


def build_llm_and_chain():
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("缺少环境变量 LLM_API_KEY，请在 .env 或系统环境中配置。")

    model_name = os.getenv("LLM_MODEL_ID", "qwen-max")
    llm = ChatTongyi(
        model=model_name,
        dashscope_api_key=api_key,
        temperature=0.2,
    )
    llm_with_tools = llm.bind_tools([fetch_financial_snapshot])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是严谨的金融与数据分析助手。涉及最新财务数据时，优先调用可用工具获取数据源说明，不得臆造未披露数字。",
        ),
        ("human", "{question}"),
    ])

    return llm_with_tools, prompt | llm_with_tools


def main() -> None:
    llm_with_tools, chain = build_llm_and_chain()

    stdout_cb = StdOutCallbackHandler()
    telemetry_cb = TelemetryMonitorHandler(trace_name="financial_query_monitor")

    question = "最新的财务数据是什么？"
    ai_msg = chain.invoke(
        {"question": question},
        config={
            "callbacks": [stdout_cb, telemetry_cb],
            "tags": ["finance", "realtime-monitoring"],
            "metadata": {"env": "dev", "module": "financial_qa"},
        },
    )

    if getattr(ai_msg, "tool_calls", None):
        tool_messages = []
        for call in ai_msg.tool_calls:
            if call["name"] == "fetch_financial_snapshot":
                tool_out = fetch_financial_snapshot.invoke(
                    call["args"],
                    config={"callbacks": [stdout_cb, telemetry_cb]},
                )
                tool_messages.append(
                    ToolMessage(content=str(tool_out), tool_call_id=call["id"])
                )

        follow_up_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是严谨的金融与数据分析助手。请基于工具返回结果给出清晰、客观的回复。",
            ),
            ("human", "{question}"),
            ai_msg,
            *tool_messages,
        ])
        final_msg = (follow_up_prompt | llm_with_tools).invoke(
            {"question": question},
            config={"callbacks": [stdout_cb, telemetry_cb]},
        )
        final_text = final_msg.content
    else:
        final_text = ai_msg.content

    print("\n👉 最终响应:\n", final_text)
    print("\n📊 结构化监控记录:")
    for item in telemetry_cb.records:
        print(" ", item)


if __name__ == "__main__":
    main()
