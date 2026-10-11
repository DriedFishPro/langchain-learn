# 第六章：Graph 高级篇

在生产环境中，节点执行可能会遇到各种意外：网络超时、API 限流、数据异常……如果任何一个错误都直接导致整个 Graph 崩溃，那系统的健壮性就太脆弱了。
LangGraph 提供了完善的容错机制，让你能对每个节点的失败行为做精细化控制。

## 6.1 重试机制
**概念：** 当节点抛出异常时，框架自动重试该节点，而不是直接崩溃。通过 `RetryPolicy` 控制重试行为。

### 6.1.1 基础重试
模拟前两次网络连接失败，第三次连接成功的场景

```python
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

    if attempt < 3:
        print(f"[retry_node] 调用失败，正在重试...", end="\n\n")
        raise ConnectionError(f"网络超时（第 {attempt} 次）")

    print(f"[retry_node] 调用成功！", end="\n\n")
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

if __name__ == "__main__":
    test_retry()
```

运行结果：
```text
[retry_node] 第 1 次调用 API（距上次 0.00s）
[retry_node] 调用失败，正在重试...

[retry_node] 第 2 次调用 API（距上次 1.19s）
[retry_node] 调用失败，正在重试...

[retry_node] 第 3 次调用 API（距上次 1.41s）
[retry_node] 调用成功！

最终结果: {'result': '成功（第 3 次尝试）'}
```


**node_attempt：感知当前是第几次尝试**
> 节点中可以通过 `runtime.execution_info.node_attempt` 获取当前是第几次执行

**RetryPolicy 的核心参数：**

| 参数             | 说明                                                 |
|----------------|----------------------------------------------------|
| `max_attempts` | 最大尝试次数（含首次执行），如 `3` 表示最多试 3 次                      |
| `retry_on`     | 可选的判断函数 `(exc) -> bool`，返回 `True` 才重试，`False` 直接抛出 |


### 6.1.2 指定异常重试
默认情况下 LangGraph 会对"大多数异常"进行重试，但实际业务中我们往往需要区分**可恢复的异常**与**不可恢复的异常**：

| 异常类型      | 举例                  | 该不该重试              |
|-----------|---------------------|--------------------|
| 可恢复的临时故障  | 网络超时、API 限流、服务暂时不可用 | ✅ 重试，过一会儿可能就通了     |
| 不可恢复的数据错误 | 参数错误、业务数据异常         | ❌ 重试一万次也是错的，应该快速失败 |

这就需要在 `RetryPolicy` 中添加 **`retry_on`** 参数，根据异常类型来决定是否重试。

```python
from langgraph.runtime import Runtime
from langgraph.graph import END, START, StateGraph

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
        .add_node("selective_retry", selective_retry_node, retry_policy=RetryPolicy(max_attempts=3, retry_on=my_retry_on))
        .add_edge(START, "selective_retry")
        .add_edge("selective_retry", END)
        .compile()
    )

    result = graph2.invoke({})
    print(f"这里不会被打印: {result}")
```

运行结果：
```text
[selective_retry_node] 第 1 次调用...
[selective_retry_node] 抛出可恢复错误

[selective_retry_node] 第 2 次调用...
[selective_retry_node] 抛出致命错误

Traceback (most recent call last):
File "D:\workspace\ai\langchain-demo\unit6_graph_高级篇\f01_graph_容错机制\f011_retry_重试机制.py", line 70, in selective_retry_node
    raise FatalException("数据损坏，无法重试")
FatalException: 数据损坏，无法重试
During task with name 'selective_retry' and id 'f43284ba-fd19-80f4-715b-ca77bf34cbd0'
```

**`retry_on` 与默认行为：**
> 如果不指定 `retry_on`，LangGraph 会对"大多数异常"进行重试。但在实际业务中，你通常只想对**可恢复的临时故障**（网络超时、限流）重试，
> 而对**不可恢复的数据错误**（空指针、数据损坏）直接失败，这就需要添加 `retry_on` 参数，根据异常类型来决定是否重试
>
> 根据文档描述，这里的大多数异常具体是指除以下（及其子类）之外的任何异常：
> ValueError、TypeError、ArithmeticError、ImportError、LookupError、NameError、SyntaxError、RuntimeError、ReferenceError、StopIteration
> StopAsyncIteration、OSError


### 6.1.3 退避策略
前面我们只关心了"要不要重试、重试几次"，但还有一个同样重要的问题：**两次重试之间要不要等？等多久？**

假设节点中调用了一个下游业务系统中的接口，此时下游接口故障了（比如接口相应503），我们的max_attempts是5，如果没有重试间隔，
比如第一次请求失败，立刻重试，又失败，继续重试...大概率重试5次后还是没能调用成功

所以退避（Backoff）策略解决的是：失败后等待一段时间再重试，并且让等待时间逐步拉长，给下游留出恢复的时间。

**退避相关参数**

| 参数                 | 说明               | 典型配置                      |
|--------------------|------------------|---------------------------|
| `initial_interval` | 首次重试前的等待时长（秒）    | 默认`0.5`秒，可根据实际场景适当加长      |
| `backoff_factor`   | 退避因子，每次等待时长乘以该系数 | 默认`2.0`（指数退避），建议保持        |
| `max_interval`     | 等待时长的上限，防止无限膨胀   | 默认`128`秒                  |
| `jitter`           | 是否给等待时长加入随机抖动    | 默认 `True`，建议保持True，防止惊群效应 |

> 实际等待时长的计算公式是：第 n 次重试等待 = `initial_interval × backoff_factor^(n-1)`，并受 `max_interval` 封顶。

**示例代码**
```python
def test_backoff():
    retry_policy = RetryPolicy(
        max_attempts=5,
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=30,
        jitter=False,
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
    test_backoff()
```

运行结果：
```text
[retry_node] 第 1 次调用 API（距上次 0.00s）
[retry_node] 调用失败，正在重试...

[retry_node] 第 2 次调用 API（距上次 1.00s）
[retry_node] 调用失败，正在重试...

[retry_node] 第 3 次调用 API（距上次 2.00s）
[retry_node] 调用失败，正在重试...

[retry_node] 第 4 次调用 API（距上次 4.00s）
[retry_node] 调用失败，正在重试...

[retry_node] 第 5 次调用 API（距上次 8.00s）
[retry_node] 调用成功！

最终结果: {'result': '成功（第 5 次尝试）'}
```
> 知识扩展：重试的节点必须要保证幂等性

---

## 6.2 超时机制
**概念：** 限制节点的最大执行时间，防止某个节点卡死拖垮整个工作流。LangGraph 提供两种超时策略，通过节点的 `timeout` 参数配置。

> **注意：超时只支持异步节点，同步节点在编译时会报错。**

### 6.2.1 run_timeout
设置节点的**总执行时长**上限，当节点执行时间超过该时长，则抛出超时异常。

```python
import asyncio
import time
from typing import TypedDict

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
        )
        .add_edge(START, "slow_node")
        .add_edge("slow_node", END)
        .compile()
    )

    result = await graph_timeout.ainvoke({})
    print(f"result={result}")

if __name__ == "__main__":
    asyncio.run(test_timeout())
```

### 6.2.2 idle_timeout
设置节点的**空闲超时时间**，当节点在运行时，超出`idle_timeout`没有捕获到"进展信号"，则抛出超时异常。

```python
import time
import asyncio
from unit1_大模型基础篇.a03_langchain调用模型 import create_model

# 模拟一个长时间运行的节点，但定期发心跳表示"还活着"
model = create_model()

async def heartbeat_node(state, runtime: Runtime):
    start_time = time.time()

    for i in range(5):
        await asyncio.sleep(1)
        # 手动触发心跳，重置 idle 时钟
        runtime.heartbeat()
        print(f"[heartbeat_node] 心跳 #{i+1} 已发送")

    # async for chunk in model.astream([HumanMessage(content="你好，请做个自我介绍")]):
    #     print(chunk.content, end="", flush=True)

    print(f"\n[heartbeat_node] 执行完成！，总耗时 {time.time() - start_time:.2f} 秒")
    return {"result": "心跳节点正常完成"}


async def test_heartbeat():
    # --- 2.3 手动心跳 runtime.heartbeat() ---
    graph_heartbeat = (
        StateGraph(State)
        .add_node("heartbeat_node", heartbeat_node, timeout=TimeoutPolicy(idle_timeout=2, refresh_on="auto"))
        .add_edge(START, "heartbeat_node")
        .add_edge("heartbeat_node", END)
        .compile()
    )

    result = await graph_heartbeat.ainvoke({})
    print(f"最终结果: {result}")

asyncio.run(test_heartbeat())
```

**进展信号：**
> refresh_on的取值有"auto"、"heartbeat"，在默认的 refresh_on="auto"下，idle时钟会在以下情况下重置：
> 1. State更新：节点中对state产生变更
> 2. 自定义流写入：在节点中通过 get_stream_writer() 向外推送自定义消息
> 3. 异步流式输出：节点中通过异步生成器（yield）向外抛出流式数据
> 4. LangChain 底层回调：如LLM 流式输出一个 Token
> 5. 子任务调度
>
> 手动调用 `runtime.heartbeat()` 也属于进度信号，与模型交互中的流式输出也属于进度信号，所以都会重置 idle 时钟，不会触发超时。
> 
> 如果refresh_on="heartbeat"，那只有手动调用 runtime.heartbeat() 才会重置idle时钟


### 6.2.3 组合使用
即使每秒都有进度信号输出，但只要总时长超过`run_timeout`还是会超时

```python
add_node("heartbeat_node", heartbeat_node, timeout=TimeoutPolicy(idle_timeout=2, run_timeout=4))
```


### 6.2.4 Send 动态超时
在上一章unit5的5.2.4 中我们学了 `Send` API 实现动态并行，`Send` 对象还支持为每个动态派发的任务**单独设置超时和重试策略**：

```python
import asyncio
import operator
from rich import print as rprint
from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import TimeoutPolicy, Send


class OverallState(TypedDict):
    tasks: list
    results: Annotated[list, operator.add]  # 多个 process_item 返回的 results 会自动拼接合并


def fan_out(state):
    sends = []

    for i, task in enumerate(state["tasks"]):
        # 不同优先级的 task 设置不同的超时
        timeout = TimeoutPolicy(run_timeout=task.get("timeout"))

        sends.append(
            Send(
                node="process_item",  # 目标节点名
                arg={"task": task, "index": i},  # 传给该节点的局部状态
                timeout=timeout,  # 动态设置超时（覆盖节点默认）
            )
        )
    print(f"[route_to_items] 扇出 {len(sends)} 个任务到 process_item")
    return sends


async def process_item(arg: dict):
    task = arg["task"]
    idx = arg["index"]
    sleep = task.get("sleep", 0.5)

    print(f"[process_item] 处理 task#{idx}（{task['name']}），预计耗时 {sleep}s")
    await asyncio.sleep(sleep)

    print(f"[process_item] {task['name']} 处理完成")
    return {"results": [{"index": idx, "status": "ok", "data": task["name"]}]}


async def test_send_timeout():
    graph = (
        StateGraph(OverallState)
        .add_node("process_item", process_item)
        .add_conditional_edges(START, fan_out)
        .add_edge("process_item", END)
        .compile()
    )

    tasks = [
        {"name": "快速任务", "priority": "high", "timeout": 1, "sleep": 0.5},
        {"name": "中速任务", "priority": "normal", "timeout": 2, "sleep": 1.5},
        {"name": "慢速任务", "priority": "high", "timeout": 3, "sleep": 3.5},
    ]

    result = await graph.ainvoke({"tasks": tasks})
    rprint(f"{result}")


if __name__ == "__main__":
    asyncio.run(test_send_timeout())
```

---

## 6.3 异常处理
重试机制为临时故障提供了弹性缓冲，超时机制则为卡死任务设定了安全边界。但在复杂的分布式业务中，当重试达到上限或超时触发强行中断时，错误最终还是会暴露出来。
如果不加处理，就会导致整个程序异常中止

### 6.3.1 error_handler
**概念：** `error_handler` 是节点的"兜底方案"。当节点重试耗尽仍然失败时，不是直接抛异常崩溃，而是调用 `error_handler` 函数进行补偿处理。

`error_handler` 接收两个参数：当前 State 和 `NodeError` 对象。返回一个 `Command` 对象，可以更新 State 并跳转到其他节点：


```python
from typing import TypedDict
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class State(TypedDict):
    order_id: str # 订单id
    amount: float  # 订单金额，用于控制支付是否成功
    order_status: str # 订单状态：paid-已支付，closed-已关闭
    close_reason: str   # 订单关闭原因

# 模拟支付节点：根据订单金额动态决定成功/失败
def payment_node(state):
    amount = state.get("amount", 0)
    print(f"[payment_node] 尝试为订单 {state.get('order_id')} 扣款，金额={amount}...")

    # 业务规则：假设用户的账户余额是1000，当订单支付金额超过1000时，支付网关返回支付失败
    user_balance = 1000

    if amount > user_balance:
        print(f"[payment_node] 支付失败：余额不足！")
        raise Exception(f"支付失败：余额不足")

    print(f"[payment_node] 支付成功！")
    return {"order_status": "paid"}

# 最终节点
def final_node(state):
    print(f"[final_node] 订单流程结束，order_status={state.get('order_status')}")
    return {}

# 支付节点的异常处理：直接在此完成订单关闭逻辑
def payment_error_handler(state: State, error: NodeError) -> Command:
    print(f"[payment_error_handler] 支付失败：{error}，关闭订单")

    return Command(
        # 直接更新为最终的“关闭”状态和原因
        update={"order_status": "closed", "close_reason": f"{error.error}"},
        goto="final_node",  # 直接跳转到最终收尾节点
    )

```

测试运行：
```python
if __name__ == "__main__":
    graph = (
        StateGraph(State)
        .add_node("payment_node", payment_node, error_handler=payment_error_handler)
        .add_node("final_node", final_node)

        .add_edge(START, "payment_node")
        .add_edge("payment_node", "final_node")
        .add_edge("final_node", END)
        .compile()
    )

    # 场景1：正常支付（金额较小，支付成功）
    result1 = graph.invoke({"order_id": "ORD-001", "amount": 200, "order_status": "pending"})
    print(f"最终结果1: {result1}")

    # 场景2：支付失败（余额不足，触发补偿：关闭订单）
    result2 = graph.invoke({"order_id": "ORD-002", "amount": 2000, "order_status": "pending"})
    print(f"最终结果2: {result2}")
```

流程解析：
```
正常流程：payment → final → END
异常流程：payment → error_handler → final → END
```

运行结果：
```text
场景1：正常支付
[payment_node] 尝试为订单 ORD-001 扣款，金额=200...
[payment_node] 支付成功！
[final_node] 订单流程结束，order_status=paid
最终结果1: {'order_id': 'ORD-001', 'amount': 200, 'order_status': 'paid'}

场景2：支付失败，触发补偿
[payment_node] 尝试为订单 ORD-002 扣款，金额=2000...
[payment_node] 支付失败：余额不足！
[payment_error_handler] 支付失败：NodeError(node='payment_node', error=Exception('支付失败：余额不足'))，关闭订单
[final_node] 订单流程结束，order_status=closed
最终结果2: {'order_id': 'ORD-002', 'amount': 2000, 'order_status': 'closed', 'close_reason': '支付失败：余额不足'}
```

**关键理解：**  
1. NodeError：`error`是原始异常信息，`node`是失败的节点名
2. Command：`error_handler` 返回的 `Command` 可以同时做两件事——更新 State 和跳转节点。
3. error_handler 与 Retry 的协作：它们不是互斥的。框架会先重试（按 `RetryPolicy`），重试耗尽后才调用 `error_handler`。如果只配了 `error_handler` 没配 `RetryPolicy`，则首次失败就直接走 `error_handler`。


### 6.3.2 set_node_defaults
如果一个 Graph 有很多节点，每个都手动配 `retry_policy`、`timeout`、`error_handler` 会很繁琐。`set_node_defaults` 
可以设置**Graph级默认配置**，所有节点自动继承

```python
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
```

测试运行：
```python
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

    result = await graph.ainvoke({"a_node_error": True})
    print(f"最终结果: {result}")


if __name__ == "__main__":
    asyncio.run(test_defaults())
```

代码解析：
> 1. 我们定义了2个异常处理器，`default_error_handler`是全局默认的，`custom_error_handler`是专门作用于b节点的
> 2. ainvoke中的step_a_error参数是为了模拟控制a节点是否要抛异常
> 3. 当a节点抛异常时，它走的是`default_error_handler`，而b节点抛异常时走的是`custom_error_handler`

运行结果：
```text
情况1：a节点异常
[a_node] 执行中...
[a_node] 模拟异常
[a_node] 执行中...
[a_node] 模拟异常
[default_error_handler] 全局处理: NodeError(node='a_node', error=ConnectionError('网络错误'))
最终结果: {'a_node_error': True, 'result': 'default-handled: 网络错误'}

情况2：b节点异常
[a_node] 执行中...
[b_node] 执行中...
[b_node] 执行中...
[custom_error_handler] 自定义处理: NodeError(node='b_node', error=Exception('数据错误'))
最终结果: {'a_node_error': False, 'result': 'custom-handled: 数据错误'}
```

**优先级规则：节点级参数 > set_node_defaults**

| 场景             | 行为           |
|----------------|--------------|
| 节点未配 + 有全局默认   | 继承全局默认       |
| 节点显式配置 + 有全局默认 | 节点级覆盖全局默认    |
| 节点未配 + 无全局默认   | 无容错策略，出错直接崩溃 |

--- 

## 6.4 中断机制（HITL）
在第四章中，我们学过 Agent 层面的 HITL（Human-in-the-Loop），在工具执行前暂停，等待人类审批。
LangGraph 提供了更底层的 **interrupt** 机制，让你在 Graph 的任意节点中暂停执行，等待人类输入后继续。

### 6.4.1 interrupt：基本使用
**概念：** `interrupt()` 是一个特殊函数，在节点内部调用时会**暂停整个 Graph 的执行**，把控制权交回给调用方。
人类做出决策后，通过 `Command(resume=...)` 恢复执行，`interrupt()` 的返回值就是 `resume` 中传入的值。

**前提条件：** interrupt 必须配合 Checkpointer 使用，因为暂停后需要靠检查点恢复执行状态。所以编译 Graph 时必须传入 checkpointer：

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    action: str # 操作选项 删库 or 跑路
    result: str # 最终结果


def request_approval_node(state: State):
    print(f"request_approval_node state: {state}")

    if state["action"] == "删库":
        # 询问是否批准，传入 True/False，interrupt的返回值就是Command(resume=approved)中传入的resume
        approved = interrupt(f"是否允许删库?")
        print(f"人工审查结果：approved={approved}")

        if approved:
            print('正在删库...')
            return Command(goto="success_node") # 批准则跳转到成功节点
        else:
            print('取消删库...')
            return Command(goto="cancel_node") # 拒绝则跳转到取消节点
    else:
        # 其他操作则直接跳转到成功节点
        return Command(goto="success_node")


def success_node(state: State):
    print(f"success_node state: {state}")
    return {"result": f"{state['action']}成功!"}


def cancel_node(state: State):
    print(f"cancel_node state: {state}")
    return {"result": f"{state['action']}失败!"}
```

测试运行：
```python
if __name__ == "__main__":
    builder = StateGraph(State)
    builder.add_node("approval", request_approval_node)
    builder.add_node("success_node", success_node)
    builder.add_node("cancel_node", cancel_node)

    builder.add_edge(START, "approval")
    builder.add_edge("success_node", END)
    builder.add_edge("cancel_node", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}

    # 第一次运行
    user_input = input("请输入你的操作（1：删库，2：跑路）").strip()
    action = '删库' if user_input == "1" else '跑路'

    result = graph.invoke({"action": action}, config)
    print("第一次运行:", result)

    # 检测是否触发了 interrupt
    if '__interrupt__' in result:
        decision = input("发现高危操作，请确认是否允许执行？(y:批准，n:拒绝): ").strip().lower()
        approved = decision == 'y'

        # 第二次运行，模拟用户审查
        result = graph.invoke(Command(resume=approved), config)
        print("第二次运行:", result)
```

**关键理解: 删库操作 interrupt 的两阶段执行：**
```
第一次 invoke:
  START → approval_node → 调用 interrupt() → 暂停
  返回：{"__interrupt__": [Interrupt(value="是否允许删库?", ...)]}

第二次 invoke(Command(resume=True)):
  恢复 → interrupt() 返回 True → 走 success_node → END
  返回：{"result": "删库成功!"}
```

> 💡 **与第四章 HITL 的对比**：第四章中 Agent 的 HITL 是通过中间件拦截工具调用实现的，只能在"工具执行前"暂停。
> 而 Graph 的 `interrupt()` 可以在节点的**任意位置**暂停，不限于工具调用——你可以在调用 LLM 前、在数据处理中途、在任何业务逻辑之间暂停。
> 这是更底层、更灵活的 HITL 机制。


### 6.4.2 审核与编辑：复杂的 HITL 交互
`interrupt` 的 `resume` 值不限于简单的 `True/False`，你可以传入任意 Python 对象（字典、列表等），实现更复杂的人机交互。

**场景：** AI 生成了 SQL `delete from user`，执行前需要人类审批。人类可以：批准执行、拒绝执行、或者**编辑修改 SQL 后再执行**。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    sql: str  # ai生成的sql
    result: str


# 模拟ai生成sql
def generate_sql(state: State):
    print("generate_sql:", state)
    sql = "delete from user"
    return {"sql": sql}


# 模拟执行sql
def execute_sql(state: State):
    print("execute_sql:", state)

    if state["sql"].startswith('delete'):
        response = interrupt(f"请求执行sql: {state['sql']}")
        print(f"interrupt response: {response}")
        action = response.get("action")

        if action == "approve":
            print("审核通过，准备执行原 SQL...")
        elif action == "reject":
            print("审核拒绝，终止执行！")
            return {"result": "SQL 被用户拒绝执行"}
        elif action == "edit":
            new_sql = response.get("sql")
            # 关键点：这里我们要把修改后的 SQL 更新到当前节点的 State 中，或者直接在这里使用它
            state["sql"] = new_sql
            print(f"用户已修改 SQL 为: {new_sql}，准备执行修改后的 SQL...")

    return {"sql": state["sql"], "result": "sql执行完成"}
```

测试运行：
```python
if __name__ == "__main__":
    builder = StateGraph(State)
    builder.add_node("generate_sql", generate_sql)
    builder.add_node("execute_sql", execute_sql)
    builder.add_edge(START, "generate_sql")
    builder.add_edge("generate_sql", "execute_sql")
    builder.add_edge("execute_sql", END)

    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "3"}}

    # 触发中断
    result = graph.invoke({"sql": "", "result": ""}, config)
    print("第一次运行:", result)

    # 检测是否触发了 interrupt
    if '__interrupt__' in result:
        decision = input("请确认是否允许执行？(y:允许，n:拒绝，e:编辑): ").strip().lower()

        if decision == 'y':
            resume = {"action": "approve"}
        elif decision == 'n':
            resume = {"action": "reject"}
        elif decision == 'e':
            # 编辑：让人类输入新的 SQL
            custom_sql = input(f"请输入修改后的 SQL: ").strip()
            # 关键设计：通过 resume 传递一个包含新 SQL 和 action 的字典
            resume = {"action": "edit", "sql": custom_sql}

        result = graph.invoke(Command(resume=resume), config)
        print("第二次运行:", result)
```

关键理解：
> 构建多字段的resume，当审核选择"编辑"时，将action和custom_sql一并返回，在恢复时就可以获取到编辑之后的SQL（custom_sql）去执行


### 6.4.3 多个并行中断
当多个并行节点同时调用 `interrupt` 时，Graph 会**一次性暂停所有中断**，在 `result["__interrupt__"]` 中返回所有中断信息。
调用方需要为每个中断构造对应的 resume 值。

```python
import operator
from typing import TypedDict, Annotated

import rich
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command, Interrupt, Send


class State(TypedDict):
    sql_list: list[str]
    execute_list: Annotated[list[str], operator.add]


def fanout(state: State):
    send_list = []

    for sql in state["sql_list"]:
        send = Send("check", sql)
        send_list.append(send)
    return send_list

def check(sql: str):
    approved = interrupt(f"请求执行sql: {sql}")
    print(f"[check] {sql} 审批结果: {approved}")

    if approved:
        return {"execute_list": [sql]}

def execute_sql(state: State):
    for sql in state["execute_list"]:
        print(f'[execute_sql] 正在执行 {sql}')
    

if __name__ == "__main__":
    builder = StateGraph(State)
    builder.add_node("check", check)
    builder.add_node("execute_sql", execute_sql)

    builder.add_conditional_edges(START, fanout, ['check'])
    builder.add_edge("check", "execute_sql")
    builder.add_edge("execute_sql", END)

    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "6"}}

    # 第一次运行，触发两个并行中断
    sql_list = [
        'update user set name="lisi" where id = 1',
        'delete from user where id = 1',
    ]

    result = graph.invoke({'sql_list': sql_list}, config)
    rich.print(result)
    print('='*100)

    # 审批流程
    interrupts = result.get("__interrupt__")
    resume_map = {} # 构造 resume 映射：{ interrupt_id: resume_value }

    for it in interrupts:
        print(it.value)
        decision = input("发现高危操作，请确认是否允许执行？(y:批准，n:拒绝): ").strip().lower()
        resume_map[it.id] = decision == 'y'

    # 第二次运行，传入映射一次性恢复所有中断
    print('=' * 100)
    final_state = graph.invoke(Command(resume=resume_map), config)
    rich.print(final_state)
```

**关键理解：**
> 每个 `Interrupt` 对象有唯一的 `id` 和 `value`（即 `interrupt()` 的参数）  
> 恢复时传 `Command(resume=resume_map)`，`resume_map` 格式必须是 `{interrupt_id: resume_value}`

---

## 6.5 本章小结
本章作为第五章 Graph 基础篇的进阶，聚焦生产环境中的三大高级能力：容错、异常处理与中断机制。

**重试机制：**
- **RetryPolicy**：通过 `max_attempts` 控制最大重试次数，`retry_on` 指定哪些异常才重试
- **node_attempt**：通过 `runtime.execution_info.node_attempt` 感知当前重试次数，实现主备切换等逻辑
- **退避策略**：通过`initial_interval`、`backoff_factor`、`max_interval`、`jitter` 这些参数来控制重试间隔

**超时机制：**
- **run_timeout**：限制节点总执行时长，超过即抛出超时异常
- **idle_timeout**：空闲超时，无进展信号即超时；通过 `runtime.heartbeat()` 手动重置 idle 时钟
- **Send 动态超时**：为动态并行任务单独设置超时策略

**异常处理：**
- **error_handler**：节点重试耗尽后的兜底方案，返回 `Command` 更新 State 并跳转到补偿节点（Saga 模式）
- **set_node_defaults**：设置图级默认容错配置，节点级参数可覆盖全局默认

**中断机制（HITL）：**
- **interrupt()**：在节点任意位置暂停 Graph 执行，等待人类输入
- **Command(resume=...)**：传入人类决策恢复执行，`resume` 值可以是任意 Python 对象
- **多个并行中断**：多个并行节点同时 `interrupt` 时，通过 `resume_map`（`{interrupt_id: resume_value}`）一次性恢复