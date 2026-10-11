import rich
from langchain.agents import AgentState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Overwrite

# 模拟总结节点
def summarize_node(state: AgentState):
    print(f"  [summarize_node] 当前消息数: {len(state['messages'])}。开始生成总结...")
    # 1. 提取需要总结的旧消息（排除最后1条最新消息）
    messages_to_summarize = state["messages"][:-1]

    # 2. 假装调用大模型生成总结
    summary = '总结：' + '；'.join([message.content for message in messages_to_summarize])
    print(f"  [summarize_node] 生成摘要: {summary}")

    # 3. 将摘要和最后一条新消息组合为新的消息列表
    messages = [SystemMessage(content=summary)] + state["messages"][-1:]

    # 4. 用新的消息列表去覆盖state中的消息列表
    return {"messages": Overwrite(messages)}

# 模拟聊天节点
def chat_node(state: AgentState):
    ai_message = AIMessage(content=f"模拟模型回复【{state["messages"][-1].content}】")
    return {"messages": [ai_message]}

# 路由函数
def before_agent(state: AgentState):
    # 路由条件：超过 5 条消息触发清理流程
    if len(state["messages"]) > 6:
        return "summarize"
    return "chat"


if __name__ == "__main__":
    builder = StateGraph(AgentState)

    builder.add_node("summarize", summarize_node)
    builder.add_node("chat", chat_node)

    builder.add_conditional_edges(START, before_agent)
    builder.add_edge("summarize", "chat")
    builder.add_edge("chat", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "001"}}

    print("\n--- 开始执行 Graph ---")
    while True:
        user_input = input("请输入：")
        result = graph.invoke(input={"messages": [HumanMessage(content=user_input)]}, config=config)
        rich.print(result)
        print('=' * 100, end='\n\n')