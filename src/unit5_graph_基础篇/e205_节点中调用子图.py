from langgraph.constants import END
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# 定义 subgraph，这些键都不与父图状态共享
class SubState(TypedDict):
    input: str
    name: str

def s_node_1(state: SubState):
    print("s_node_1")
    return {"name": "sub graph"}

def s_node_2(state: SubState):
    print("s_node_2")
    return {"input": state["input"] + '-' + state["name"]}

subgraph_workflow = StateGraph(SubState)
subgraph_workflow.add_node(s_node_1)
subgraph_workflow.add_node(s_node_2)
subgraph_workflow.add_edge(START, "s_node_1")
subgraph_workflow.add_edge("s_node_1", "s_node_2")
subgraph_workflow.add_edge("s_node_2", END)
subgraph = subgraph_workflow.compile()

class ParentState(TypedDict):
    input: str
    name: str

def p_node_1(state: ParentState):
    print("p_node_1")
    return {"name": 'parent graph'}

def p_node_2(state: ParentState):
    # 将状态转换为子图状态
    print("p_node_2")
    sub_state = subgraph.invoke({"input": state["input"]})
    print(f'sub_state：{sub_state}')
    return {"input": state["input"] + ' ' + state["name"]}

workflow = StateGraph(ParentState)
workflow.add_node("p_node_1", p_node_1)
workflow.add_node("p_node_2", p_node_2)
workflow.add_edge(START, "p_node_1")
workflow.add_edge("p_node_1", "p_node_2")
workflow.add_edge("p_node_2", END)
graph = workflow.compile()

parent_state = graph.invoke({"input": "hello world"})
print(f'parent_state：{parent_state}')