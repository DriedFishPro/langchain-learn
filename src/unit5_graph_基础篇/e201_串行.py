import os
import rich

from pathlib import Path
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit5_graph_基础篇.e101_graph_快速上手 import draw_graph


class State(TypedDict):
    file: str   # 文件名称
    check: bool # 文件类型检查结果
    content: str # 文件内容
    summary: str # 模型总结内容

# 节点1：检查文件类型
def check_file(state: State):
    print(f"[check_file] file: {state['file']}")
    path = Path(state['file'])
    return {"check": path.is_file() and path.suffix == ".py"}

# 节点2：读取文件内容
def read_file(state: State):
    print(f"[read_file] file: {state['file']}")
    content = Path(state['file']).read_text(encoding="utf-8")
    print(f"[read_file] content：{content[:100]}", end="\n\n")
    return {"content": content}

# 节点3：调用大模型对文件内容做总结
def llm_summary(state: State):
    model = create_model()
    msg = model.invoke(f"对这段代码逻辑做一个简单总结，字数控制在500以内：{state['content']}")
    print(f"[llm_summary] 总结: {msg.content}", end="\n\n")
    return {"summary": msg.content}

# 路由函数
def check_router(state: State):
    if state["check"]:
        return "read"
    return "end"


# 构建工作流
workflow = StateGraph(State)

workflow.add_node("check_file", check_file)
workflow.add_node("read_file", read_file)
workflow.add_node("llm_summary", llm_summary)

workflow.add_edge(START, "check_file")
workflow.add_conditional_edges(
    "check_file", check_router, {"read": "read_file", "end": END}
)
workflow.add_edge("read_file", "llm_summary")
workflow.add_edge("llm_summary", END)

graph = workflow.compile()
current = Path(__file__).resolve()
draw_graph(graph, (current.parent / "e021_串行.png"))

# 开始运行
state = graph.invoke({"file": current})
rich.print(state)