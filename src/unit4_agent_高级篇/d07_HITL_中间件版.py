import rich
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit3_agent_基础篇.c06_text2sql import get_db_schema, execute_sql, get_text2sql_prompt


def should_interrupt(request: ToolCallRequest) -> bool:
    args = request.tool_call.get("args", {})
    query = args.get("query", '')
    # select语句直接放行，非select语句拦截
    return not query.lower().startswith('select')


def hitl_middleware():
    print("\n[DEBUG] 初始化 HITL 中间件...")
    middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "execute_sql": {
                "allowed_decisions": ["approve", "edit", "reject", "respond"],
                "when": should_interrupt,  # 核心：在这里绑定你的条件拦截函数
            },
        },
        description_prefix="等待人工确认",
    )
    print("[DEBUG] HITL 中间件初始化完成\n")
    return middleware


if __name__ == "__main__":
    hitl_prompt = '注意事项：当你发现execute_sql工具执行时的实际参数与你事先规划的不一致时，表示被人工审核修改过了，请接受该结果作为最终状态，不要尝试重复执行或纠正'

    agent = create_agent(
        model=create_model(enable_thinking=True),
        tools=[get_db_schema, execute_sql],
        middleware=[hitl_middleware()],
        checkpointer=InMemorySaver(),
        system_prompt=get_text2sql_prompt() + '\n' + hitl_prompt,
    )

    config = {"configurable": {"thread_id": "1"}}
    print("=== sql助手已启动 ===")

    while True: # 外层是连续对话的循环
        user_input = input("你: ").strip()
        current_input = {"messages": [{"role": "user", "content": user_input}]}

        while True: # 内层是agent的事件循环
            result = agent.invoke(current_input, config=config)

            print(f"\n{'=' * 100} 消息列表 ↓↓↓ {'=' * 100}")
            for index, message in enumerate(result['messages']):
                print(f'{index + 1} {type(message)}\n{message}', end='\n\n')
            print(f"{'=' * 100} 消息列表 ↑↑↑ {'=' * 100}\n")

            if '__interrupt__' in result:
                rich.print(result['__interrupt__'])
                print('=' * 100)
                decisions = []  # 1. 创建一个空列表，用来收集所有 tool_call 的处理结果

                # 遍历所有中断对象
                for interrupt in result['__interrupt__']:
                    action_requests = interrupt.value.get('action_requests', [])

                    # 逐个询问用户并收集决策
                    for idx, action in enumerate(action_requests, 1):
                        print(f"\n--------------------------------------------------")
                        print(f"检测到高危操作 {interrupt.id} [{idx}/{len(action_requests)}]")
                        print(action['description'])
                        print(f"--------------------------------------------------")

                        tool_name = action.get('name')
                        tool_args = action.get('args')
                        decision = input("请选择 (y:批准，n:拒绝，e:编辑，r:模拟回复): ").strip().lower()

                        if decision == "y":
                            print(f"-> 操作 [{idx}] 已批准")
                            decisions.append({"type": "approve"})

                        elif decision == "e":
                            input_sql = input(f"-> 请输入操作 [{idx}] 修改后的 SQL: ").strip()
                            decisions.append({
                                "type": "edit",
                                "edited_action": {
                                    "name": tool_name,
                                    "args": {"query": input_sql}  # 保留原工具名，仅替换修改后的参数
                                }
                            })

                        elif decision == "n":
                            print(f"-> 操作 [{idx}] 已拒绝")
                            decisions.append({
                                "type": "reject",
                                "message": f"系统反馈：用户拒绝执行工具 [{tool_name}]"
                            })

                        elif decision == "r":
                            custom_resp = input(f"-> 请输入操作 [{idx}] 拟造的返回结果(留空使用默认): ").strip()
                            decisions.append({
                                "type": "respond",
                                "message": custom_resp or "系统反馈：模拟工具执行成功（实际数据未修改）"
                            })

                # 统一打包提交，处理完成后继续循环，下一次invoke时，会使用 Command 恢复
                current_input = Command(resume={"decisions": decisions})
            else:
                # 没有中断，说明任务已完成，打印结果并跳出循环等待下一次用户输入
                print(f"sql助手: {result['messages'][-1].content}", end="\n\n")
                break