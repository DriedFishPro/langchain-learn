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