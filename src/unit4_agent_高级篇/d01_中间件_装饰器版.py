from dataclasses import dataclass
from langchain.agents.middleware import before_model, after_model, wrap_tool_call, after_agent, before_agent
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from typing import Any
from collections.abc import Callable

from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit3_agent_基础篇.c02_agent_tool import get_location_mock, get_weather2


@dataclass
class UserContext:
    user_id: str
    user_token: int  # 用户剩余 token 额度


@before_agent
def before_agent_log(state: AgentState, runtime: Runtime) -> None:
    print("****************************** before_agent ******************************")
    print(f"user context: {runtime.context}，当前会话消息数: {len(state["messages"])}", end="\n\n")

@after_agent
def after_agent_log(state: AgentState, runtime: Runtime) -> None:
    print("****************************** after_agent ******************************")
    print(f"user context: {runtime.context}，当前会话消息数: {len(state["messages"])}", end="\n\n")


@before_model(can_jump_to=["end"])
def check_user_token(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("****************************** before_model ******************************")
    print(f"当前用户剩余token：{runtime.context.user_token}，当前会话消息数: {len(state["messages"])}", end="\n\n")

    if runtime.context.user_token <= 0:
        return {"messages": [AIMessage("剩余token不足，无法继续对话")], "jump_to": "end"}
    return None


@after_model
def calc_user_token(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("****************************** after_model ******************************")
    # 拿到最新的一条模型返回消息
    last_msg = state['messages'][-1]

    # 健壮性检查：确保消息包含 usage_metadata (有些本地小模型可能不返回这个字段)
    if hasattr(last_msg, 'usage_metadata') and last_msg.usage_metadata:
        usage_token = last_msg.usage_metadata.get('total_tokens', 0)
        # 核心逻辑：扣除用户的 token
        runtime.context.user_token -= usage_token
        print(f"当前用户剩余token: {runtime.context.user_token}，当前会话消息数: {len(state["messages"])}，本次消耗: {usage_token} token")
    else:
        print("未获取到 token 消耗数据。")

    print("\n", end="")
    return None


@wrap_tool_call
def monitor_tool(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
    print("****************************** tool_call_wrapper ******************************")
    print(f"调用工具: {request.tool_call['name']}，参数: {request.tool_call['args']}")

    try:
        result = handler(request)
        print(f"工具结果={result}", end="\n\n")
        return result
    except Exception as e:
        print(f"Tool failed: {e}")
        raise


if __name__ == "__main__":
    # 组装 Agent，把我们的数据库的两个工具塞进去
    agent = create_agent(
        model=create_model(),
        tools=[get_location_mock, get_weather2],
        middleware=[before_agent_log, after_agent_log, check_user_token, calc_user_token, monitor_tool],
        checkpointer=InMemorySaver(),
        context_schema=UserContext,
        system_prompt="你是一个天气预查询助手",
    )

    config = {"configurable": {"thread_id": "1"}}
    user_context = UserContext(user_id="user_123", user_token=2000)

    print("=== 天气预查询助手已启动 ===")
    print("输入 'quit' 或 'exit' 退出对话\n")

    while True:
        user_input = input("你: ").strip()

        if not user_input:  continue

        if user_input.lower() in ['quit', 'exit']:
            print("再见！")
            break

        messages = {"messages": [{"role": "user", "content": user_input}]}
        response = agent.invoke(input=messages, config=config, context=user_context)
        print(response['messages'][-1].content, end="\n\n")