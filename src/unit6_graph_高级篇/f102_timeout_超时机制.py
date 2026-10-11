import time
import asyncio
from typing import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.errors import NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import TimeoutPolicy, RetryPolicy
from unit1_大模型基础篇.a03_langchain调用模型 import create_model

class State(TypedDict):
    result: str

async def slow_node(state, runtime: Runtime):
    attempt = runtime.execution_info.node_attempt

    if attempt == 1:
        print("[slow_node] 第一次执行，模拟接口阻塞，睡眠 5 秒...")
        await asyncio.sleep(5)
        return {"result": "看不到我"}
    else:
        print("[slow_node] 重试执行，模拟切换备用接口")
        return {"result": "备用接口调用成功"}


async def test_timeout():
    graph_timeout = (
        StateGraph(State)
        .add_node("slow_node", slow_node,
            timeout=TimeoutPolicy(run_timeout=3),
            retry_policy=RetryPolicy(max_attempts=3, retry_on=NodeTimeoutError)
        )  # 总执行时长上限：2秒总超时
        .add_edge(START, "slow_node")
        .add_edge("slow_node", END)
        .compile()
    )

    result = await graph_timeout.ainvoke({})
    print(f"result={result}")


# 模拟一个长时间运行的节点，但定期发心跳表示"还活着"
model = create_model()

async def heartbeat_node(state, runtime: Runtime):
    start_time = time.time()

    for i in range(5):
        await asyncio.sleep(1)
        # 手动触发心跳，重置 idle 时钟
        runtime.heartbeat()
        print(f"[heartbeat_node] 心跳 #{i+1} 已发送")

    # async for chunk in model.astream([HumanMessage(content="简单介绍一下langchain是什么？")]):
    #     print(chunk.content, end="", flush=True)

    print(f"\n[heartbeat_node] 执行完成！，总耗时 {time.time() - start_time:.2f} 秒")
    return {"result": "心跳节点正常完成"}


async def test_heartbeat():
    # --- 2.3 手动心跳 runtime.heartbeat() ---
    graph_heartbeat = (
        StateGraph(State)
        .add_node("heartbeat_node", heartbeat_node, timeout=TimeoutPolicy(idle_timeout=2, run_timeout=4, refresh_on="auto"))
        .add_edge(START, "heartbeat_node")
        .add_edge("heartbeat_node", END)
        .compile()
    )

    result = await graph_heartbeat.ainvoke({})
    print(f"最终结果: {result}")


if __name__ == "__main__":
    asyncio.run(test_timeout())
    # asyncio.run(test_heartbeat())