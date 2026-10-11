from langgraph.constants import END
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class SubState(TypedDict):
    input: str
    name: str

def s_node_1(state: SubState):
    print(f"s_node_1 state: {state}")
    return {"name": "sub graph"}

def s_node_2(state: SubState):
    # 这个节点使用子图私有的 'bar'，然后更新共享的 'foo'
    print(f's_node_2: state: {state}')
    return {"input": state["input"] + ' ' + state["name"]}

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
    print(f"p_node_1 state: {state}")
    return {"name": 'parent graph'}

workflow = StateGraph(ParentState)
workflow.add_node("p_node_1", p_node_1)
workflow.add_node("p_node_2", subgraph)  # 直接把子图当节点

workflow.add_edge(START, "p_node_1")
workflow.add_edge("p_node_1", "p_node_2")
workflow.add_edge("p_node_2", END)
graph = workflow.compile()

parent_state = graph.invoke({"input": "hello!"})
print(f'parent_state：{parent_state}')