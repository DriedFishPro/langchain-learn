import operator
import os
import rich
from langchain.messages import AnyMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langchain.messages import SystemMessage, ToolMessage, HumanMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END
from unit1_大模型基础篇.a03_langchain调用模型 import create_model

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    tool_calls: int

@tool(parse_docstring=True)
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool(parse_docstring=True)
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool(parse_docstring=True)
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

tools = [add, multiply, divide]
tools_map = {tool.name: tool for tool in tools}


def llm_node(state: MessagesState):
    # 1. 取出当前对话历史
    messages = state["messages"]
    print('llm_node被执行')

    model = create_model()
    model_with_tools = model.bind_tools(tools)

    # 2. 拼接系统提示词 + 用户对话历史，调用大模型，拿到回复
    system_prompt = SystemMessage(content="你是一个乐于助人的助手，你可以帮助用户做一些数学运算。")
    ai_message = model_with_tools.invoke([system_prompt] + messages)
    # 3. 累计 LLM 调用次数
    call_count = state.get("llm_calls", 0) + 1
    # 4. 返回状态更新
    return {"messages": [ai_message], "llm_calls": call_count}

def tool_node(state: MessagesState):
    tool_messages = []
    tool_calls = state.get("tool_calls", 0)
    print(f'tool_node toll_calls: {state["messages"][-1].tool_calls}')

    for tool_call in state["messages"][-1].tool_calls:
        tool_calls += 1
        tool = tools_map[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {"messages": tool_messages, "tool_calls": tool_calls}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


def draw_graph(agent, output_path: str):
    graph_png = agent.get_graph(xray=True).draw_mermaid_png()

    with open(output_path, "wb") as f:
        f.write(graph_png)

    print(f"Agent graph 已保存到: {output_path}")

if __name__ == '__main__':
    graph_builder = StateGraph(MessagesState)

    # 注册节点
    graph_builder.add_node("llm_node", llm_node)
    graph_builder.add_node("tool_node", tool_node)

    # 连接边
    graph_builder.add_edge(START, "llm_node")                          # 起点 → llm_node
    graph_builder.add_conditional_edges("llm_node", should_continue)   # llm_node → 条件路由
    graph_builder.add_edge("tool_node", "llm_node")                    # tool_node → llm_node（形成循环）

    graph = graph_builder.compile()
    # 画出流程图
    # output = os.path.join(os.path.dirname(__file__), "react_graph.png")
    # draw_graph(graph, output)

    # 构建初始state
    input_state = {
        "messages": [HumanMessage(content="123加210，再乘2等于几？")],
        # "llm_calls": 0,
        # "tool_calls": 0
    }

    # Step 7: 把state对象传入invoke，运行graph程序
    result_state = graph.invoke(input=input_state)
    rich.print(result_state)