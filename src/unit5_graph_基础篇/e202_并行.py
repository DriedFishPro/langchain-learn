import rich
from pathlib import Path
from typing import TypedDict
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit5_graph_基础篇.e101_graph_快速上手 import draw_graph


class State(TypedDict):
    topic: str # 主题
    joke: str   # 笑话
    story: str  # 故事
    poem: str   # 诗歌
    combined_output: str # 合并


model = create_model()

# 第一个LLM调用生成初始笑话
def call_llm_1(state: State):
    print(f"[call_llm_1] 生成笑话: {state['topic']}")
    msg = model.invoke(f"写一个关于{state['topic']}的小笑话(200字以内)")
    print(f"  [call_llm_1] 笑话: {msg.content}", end="\n\n")
    return {"joke": msg.content}


# 第二个LLM调用生成故事
def call_llm_2(state: State):
    print(f"[call_llm_2] 创建故事: {state['topic']}")
    msg = model.invoke(f"写一个关于{state['topic']}的小故事(200字以内)")
    print(f"  [call_llm_2] 故事: {msg.content}", end="\n\n")
    return {"story": msg.content}


# 第三个LLM的呼叫生成诗歌
def call_llm_3(state: State):
    print(f"[call_llm_3] 创建诗歌: {state['topic']}")
    msg = model.invoke(f"写一首关于{state['topic']}的诗(200字以内)")
    print(f"  [call_llm_3] 诗歌: {msg.content}", end="\n\n")
    return {"poem": msg.content}


# 将笑话、故事和诗歌合并成一个输出
def aggregator(state: State):
    combined = f"这里有一个关于{state['topic']}的故事、笑话和诗歌！\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}


workflow = StateGraph(State)

workflow.add_node("call_llm_1", call_llm_1)
workflow.add_node("call_llm_2", call_llm_2)
workflow.add_node("call_llm_3", call_llm_3)
workflow.add_node("aggregator", aggregator)

workflow.add_edge(START, "call_llm_1")
workflow.add_edge(START, "call_llm_2")
workflow.add_edge(START, "call_llm_3")
workflow.add_edge("call_llm_1", "aggregator")
workflow.add_edge("call_llm_2", "aggregator")
workflow.add_edge("call_llm_3", "aggregator")
workflow.add_edge("aggregator", END)

graph = workflow.compile()
draw_graph(graph, (Path(__file__).resolve().parent / "e022_并行.png"))

result = graph.invoke({"topic": "猫"})
rich.print(result)