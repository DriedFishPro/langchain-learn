import rich
from typing import TypedDict, Literal, Annotated, Callable
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import OmitFromInput, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage, AIMessage, SystemMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from typing_extensions import NotRequired

from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit3_agent_基础篇.c02_agent_tool import get_location_mock, get_weather2


class TodoItem(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]

class TodoListState(TypedDict):
    # 扩展字段：待办列表，用 OmitFromInput 标记为"不作为用户输入"
    todos: NotRequired[list[TodoItem] | None]


@tool(description="更新任务列表")
def write_todos(todos: list[TodoItem], tool_call_id: Annotated[str, InjectedToolCallId]):
    print(f"write_todos 被调用：todos={todos}")
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"任务列表更新为：{todos}", tool_call_id=tool_call_id)],
        }
    )


class TodoListMiddleware(AgentMiddleware):
    state_schema = TodoListState   # 声明状态扩展
    tools = [write_todos]

    planning_prompt = '''
    # 工作流程：
    当遇到复杂任务（至少要3个步骤才能完成的任务）时，或者用户一次提出多个任务时，请务必先做规划：

    1. 先把大任务拆解为若干个结构化的任务列表，并调用 `write_todos` 工具来写下待办任务列表，任务的初始状态为 "pending"
    2. 每个任务开始之前，先调用 `write_todos` 工具将该任务的状态改为 "in_progress"
    3. 每个任务完成时，调用 `write_todos` 工具将该任务的状态改为 "completed"
    '''

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
        print(f'system_message：{request.system_message}')
        system_prompt = request.system_message.content + '\n' + self.planning_prompt
        system_message = SystemMessage(content=system_prompt)
        return handler(request.override(system_message=system_message))


if __name__ == "__main__":
    agent = create_agent(
        model=create_model(enable_thinking=True),
        tools=[get_location_mock, get_weather2],
        middleware=[TodoListMiddleware()],
        system_prompt="你是一个天气预查询助手",
    )

    # 查看合并后的完整 State 属性列表
    rich.print(agent.builder.schemas)
    print('-' * 100)
    # 查看编译后的 LangGraph 图中包含的所有工具列表
    print(agent.get_graph().nodes.get('tools'), end='\n\n')

    messages = {"messages": [{"role": "user", "content": "帮我查一下，今天我这里的天气和上海的天气情况，看看差别大不大"}]}
    response = agent.invoke(input=messages)
    rich.print(response['messages'])