import asyncio
from typing import TypedDict
from langgraph.constants import END, START
from langgraph.errors import NodeError
from langgraph.graph import StateGraph
from langgraph.types import Command, RetryPolicy, TimeoutPolicy


class State(TypedDict):
    a_node_error: bool # 控制节点a要不要抛异常
    result: str


async def a_node(state):
    print(f"[a_node] 执行中...")

    if state.get("a_node_error"):
        print("[a_node] 模拟异常")
        raise ConnectionError("网络错误")

async def b_node(state):
    print("[b_node] 执行中...")
    raise Exception("数据错误")


async def default_error_handler(state: State, error: NodeError) -> Command:
    print(f"[default_error_handler] 全局处理: {error}")
    return Command(update={"result": f"default-handled: {error.error}"}, goto=END)

async def custom_error_handler(state: State, error: NodeError) -> Command:
    print(f"[custom_error_handler] 自定义处理: {error}")
    return Command(update={"result": f"custom-handled: {error.error}"}, goto=END)

async def test_defaults():
    graph = (
        StateGraph(State)
        .set_node_defaults(
            retry_policy=RetryPolicy(max_attempts=2),
            error_handler=default_error_handler
        )
        .add_node("a_node", a_node)  # ← 继承全局默认
        .add_node("b_node", b_node, error_handler=custom_error_handler)  # ← 覆盖全局！
        .add_edge(START, "a_node")
        .add_edge("a_node", "b_node")
        .add_edge("b_node", END)
        .compile()
    )

    result = await graph.ainvoke({"a_node_error": False})
    print(f"最终结果: {result}")


if __name__ == "__main__":
    asyncio.run(test_defaults())