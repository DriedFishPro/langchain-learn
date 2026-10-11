import time
from typing import TypedDict
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    result: str

# 模拟前两次调用都抛 ConnectionError，第三次才成功
last_time = time.time()
def retry_node(state, runtime: Runtime):
    attempt = runtime.execution_info.node_attempt

    now = time.time()
    global last_time
    print(f"[retry_node] 第 {attempt} 次调用 API（距上次 {now - last_time:.2f}s）")
    last_time = now

    if attempt < 5:
        print(f"[retry_node] 第 {attempt} 次调用失败，正在重试...")
        raise ConnectionError(f"网络超时（第 {attempt} 次）")

    print(f"[retry_node] 第 {attempt} 次调用成功！")
    return {"result": f"成功（第 {attempt} 次尝试）"}

def test_retry():
    graph1 = (
        StateGraph(State)
        .add_node("retry_node", retry_node, retry_policy=RetryPolicy(max_attempts=3))
        .add_edge(START, "retry_node")
        .add_edge("retry_node", END)
        .compile()
    )

    result = graph1.invoke({})
    print(f"最终结果: {result}")


# 可恢复的异常
class RecoverableException(Exception):
    pass

# 不可恢复的异常
class FatalException(Exception):
    pass

# 根据异常类型判断，只在 RecoverableException 时重试
def my_retry_on(exc: BaseException) -> bool:
    if isinstance(exc, RecoverableException):
        return True
    return False

# 模拟抛 RecoverableException 可重试，抛 FatalException 不可重试
def selective_retry_node(state, runtime: Runtime):
    attempt = runtime.execution_info.node_attempt
    print(f"[selective_retry_node] 第 {attempt} 次调用...")

    if attempt == 1:
        # 第一次抛可恢复错误 —— 应该重试
        print(f"[selective_retry_node] 抛出可恢复错误")
        raise RecoverableException("临时故障，可以重试")

    # 第二次抛致命错误 —— 不应该重试，直接失败
    print(f"[selective_retry_node] 抛出致命错误")
    raise FatalException("数据损坏，无法重试")

def test_selective_retry():
    print("\n--- 1.2 自定义 retry_on：只在 RecoverableException 时重试 ---")
    graph2 = (
        StateGraph(State)
        .add_node("selective_retry", selective_retry_node,
                  retry_policy=RetryPolicy(max_attempts=3, retry_on=my_retry_on))
        .add_edge(START, "selective_retry")
        .add_edge("selective_retry", END)
        .compile()
    )

    result = graph2.invoke({})
    print(f"这里不会被打印: {result}")


def test_backoff():
    retry_policy = RetryPolicy(
        max_attempts=5,
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=30,
        # jitter=False,
    )

    graph1 = (
        StateGraph(State)
        .add_node("retry_node", retry_node, retry_policy=retry_policy)
        .add_edge(START, "retry_node")
        .add_edge("retry_node", END)
        .compile()
    )

    result = graph1.invoke({})
    print(f"最终结果: {result}")


if __name__ == "__main__":
    test_retry()
    # test_selective_retry()
    # test_backoff()