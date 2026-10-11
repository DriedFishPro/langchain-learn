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