import rich
from pathlib import Path
from typing import TypedDict, Literal
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langchain.messages import HumanMessage, SystemMessage
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit5_graph_基础篇.e101_graph_快速上手 import draw_graph


class State(TypedDict):
    input: str      # 用户问题
    intention: str   # 意图类型（story-讲故事，joke-讲笑话，poem-写诗歌）
    output: str     # 最终结果

model = create_model()

# 意图分析
def llm_intention(state: State):
    messages = [
        SystemMessage(content="分析用户输入的意图，如果是讲故事，则输出 story；如果是讲笑话，则输出 joke；如果是写诗歌，则输出poem；只能输出这三个词之一，不要输出任何其他内容"),
        HumanMessage(content=state["input"]),
    ]

    res = model.invoke(messages)
    print(f"[llm_call_router] 意图分析结果: {res.content}")
    return {"intention": res.content}

# 写故事
def llm_call_1(state: State):
    print(f"[llm_call_1] 创建故事: {state['input']}")
    result = model.invoke(state["input"])
    print(f"  [llm_call_1] 故事: {result.content}", end="\n\n")
    return {"output": result.content}

# 写笑话
def llm_call_2(state: State):
    print(f"[llm_call_2] 创建笑话: {state['input']}")
    result = model.invoke(state["input"])
    print(f"  [llm_call_2] 笑话: {result.content}", end="\n\n")
    return {"output": result.content}

# 写首诗
def llm_call_3(state: State):
    print(f"[llm_call_3] 创建诗歌: {state['input']}")
    result = model.invoke(state["input"])
    print(f"  [llm_call_3] 诗歌: {result.content}", end="\n\n")
    return {"output": result.content}

# 路由函数
def intention_router(state: State) -> Literal["llm_call_1", "llm_call_2", "llm_call_3"]:
    # 返回接下来要访问的节点名称
    if state["intention"] == "story":
        return "llm_call_1"
    elif state["intention"] == "joke":
        return "llm_call_2"
    elif state["intention"] == "poem":
        return "llm_call_3"


workflow = StateGraph(State)

workflow.add_node("llm_intention", llm_intention)
workflow.add_node("llm_call_1", llm_call_1)
workflow.add_node("llm_call_2", llm_call_2)
workflow.add_node("llm_call_3", llm_call_3)

workflow.add_edge(START, "llm_intention")
workflow.add_conditional_edges("llm_intention", intention_router)
workflow.add_edge("llm_call_1", END)
workflow.add_edge("llm_call_2", END)
workflow.add_edge("llm_call_3", END)

graph = workflow.compile()
# draw_graph(graph, (Path(__file__).resolve().parent / "e023_路由.png"))

result = graph.invoke({"input": "给我讲一个关于猫的笑话"})
rich.print(result)