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
    print('=' * 100)

    # 审批流程
    interrupts = result.get("__interrupt__")
    resume_map = {}  # 构造 resume 映射：{ interrupt_id: resume_value }

    for it in interrupts:
        print(it.value)
        decision = input("发现高危操作，请确认是否允许执行？(y:批准，n:拒绝): ").strip().lower()
        resume_map[it.id] = decision == 'y'

    # 第二次运行，传入映射一次性恢复所有中断
    print('=' * 100)
    final_state = graph.invoke(Command(resume=resume_map), config)
    rich.print(final_state)