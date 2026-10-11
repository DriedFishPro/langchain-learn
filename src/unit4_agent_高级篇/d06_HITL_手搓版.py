from collections.abc import Callable
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit3_agent_基础篇.c06_text2sql import get_db_schema, execute_sql, get_text2sql_prompt


class HumanApprovalRequiredException(Exception):
    def __init__(self, tool_name: str, tool_args: dict):
        self.tool_name = tool_name
        self.tool_args = tool_args

@dataclass
class UserContext:
    approved: bool # 是否允许执行高危操作


@wrap_tool_call
def execute_sql_wrap(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
    print("****************************** wrap_tool_call ******************************")
    print(f"调用工具: {request.tool_call['name']}，参数: {request.tool_call['args']}")

    if request.tool_call["name"] == "execute_sql":
        if (not request.tool_call["args"]["query"].lower().startswith("select")
                and not request.runtime.context.approved
        ):
            raise HumanApprovalRequiredException(tool_name=request.tool_call["name"], tool_args=request.tool_call["args"])

    result = handler(request)
    print(f"工具结果={result}", end="\n\n")
    return result


if __name__ == "__main__":
    agent = create_agent(
        model=create_model(enable_thinking=True),
        tools=[get_db_schema, execute_sql],
        middleware=[execute_sql_wrap],
        checkpointer=InMemorySaver(),
        context_schema=UserContext,
        system_prompt=get_text2sql_prompt(),
    )

    config = {"configurable": {"thread_id": "1"}}
    user_context = UserContext(approved=False)
    print("=== sql助手已启动 ===")

    while True:
        user_input = input("你: ").strip()

        try:
            messages = {"messages": [{"role": "user", "content": user_input}]}
            result = agent.invoke(input=messages, config=config, context=user_context, )

            print('-' * 100)
            for index, message in enumerate(result['messages']):
                print(f'{index + 1} {type(message)}\n{message}', end='\n\n')
            print('-' * 100)

            print(result['messages'][-1].content)
        except HumanApprovalRequiredException as e:
            # 捕获到中间件抛出的异常，进入人工干预流程
            print("\n" + "!" * 40)
            print(f"🚨 [安全拦截] 检测到高危 SQL 操作")
            print(f"📦 拟执行语句: {e.tool_args['query']}")
            print("!" * 40)
            decision = input("\n请确认是否允许执行此删除操作？(y：批准，n：拒绝): ").strip().lower()

            if decision == "y":
                print("\n✅ 已批准，正在恢复执行...")
                user_context.approved = True
                # 核心 1：恢复执行，传入 input=None，agent 会从断点处继续运行
                result = agent.invoke(input=None, config=config, context=user_context)
                # 执行完毕后，务必将权限重置为 False，防止后续提权越界
                user_context.approved = False
            else:
                print("\n❌ 已拒绝。正在将拒绝原因反馈给大模型...")
                user_context.approved = False

                # 核心 2：获取当前被挂起的state
                current_state = agent.get_state(config)
                last_message = current_state.values["messages"][-1]

                # 提取出被拦截的 tool_call_id
                tool_call_id = None
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    for tc in last_message.tool_calls:
                        if tc["name"] == e.tool_name:
                            tool_call_id = tc["id"]
                            break

                if tool_call_id:
                    # 核心 3：伪造一条 ToolMessage，告知大模型操作已被拒绝
                    reject_msg = ToolMessage(content="系统反馈：用户拒绝了本次工具调用", tool_call_id=tool_call_id)

                    # 核心 4：把ToolMessage塞到state中，as_node="tools" 表示将这个 reject_msg 作为工具节点的输出写入state
                    agent.update_state(config=config, values={"messages": [reject_msg]}, as_node="tools")

                    # 核心 5：状态修改完毕后，重新驱动。此时工具节点已被我们手动跳过，大模型会直接拿到拒绝反馈
                    result = agent.invoke(input=None, config=config, context=user_context)

            print('=' * 100)
            for index, message in enumerate(result['messages']):
                print(f'{index + 1} {type(message)}\n{message}', end='\n\n')
            print('=' * 100)