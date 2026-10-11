import sqlite3
import uuid
from typing import Any
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent, AgentState
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from langgraph.runtime import Runtime
from langchain_core.messages import RemoveMessage, HumanMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model, SummarizationMiddleware
from unit3_agent_基础篇.c06_text2sql import get_db_schema, execute_sql, get_text2sql_prompt


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """上下文裁剪：当消息过长时触发裁剪，防止超出上下文窗口"""
    messages = state["messages"]
    print("before_model 当前会话消息数:", len(messages))

    # 1. 找出所有 HumanMessage 的索引，每条 HumanMessage 标志一轮对话的开始
    human_indices = [i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)]
    num_rounds = len(human_indices)

    # 2. 轮次不超过 3 时不裁剪
    if num_rounds < 3: return None

    print("****************************** 触发裁剪 ******************************")
    # 3. 设置需要保留的轮数（最近 n 轮）
    n = 1

    # 4. 确定裁剪起始位置：从倒数第 n 条 HumanMessage 开始截取
    target_idx = human_indices[-n]

    # 5. 截取最近 n 轮完整对话（包含中间的 AIMessage、ToolMessage 等）
    recent_messages = messages[target_idx:]

    print(f"原始对话轮数：{num_rounds}，保留最近 {n} 轮，裁剪后消息数：{len(recent_messages)}")
    # 6. 删除旧消息并写入裁剪后的消息列表
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_messages]}


if __name__ == "__main__":
    summarization = SummarizationMiddleware(
        model=create_model(model='qwen3.6-flash'),
        trigger=("tokens", 2500),  # 上下文中超过2000个token时，触发总结
        keep=("messages", 4),
        # summary_prompt='自定义总结提示词'
    )

    # 创建模型
    model = create_model(enable_thinking=True)

    # 创建短期记忆 基于 SQLite
    conn = sqlite3.connect("./sqlite/agent_memory.db", check_same_thread=False) # check_same_thread=False 允许跨线程复用连接
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()        # 初始化数据库

    agent = create_agent(
        model=model,
        tools=[get_db_schema, execute_sql],
        checkpointer=checkpointer,
        middleware=[summarization],
        system_prompt=get_text2sql_prompt(),
    )

    # 使用 thread_id 来追踪会话
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print(f"config={config}，输入 'quit' 或 'exit' 退出对话\n")

    while True:
        user_input = input("你: ").strip()
        if user_input == "quit": break
        if not user_input.strip(): continue

        messages = {"messages": [{"role": "user", "content": user_input}]}
        response = agent.invoke(input=messages, config=config)

        print(f'总消息数：{len(response['messages'])}，usage_metadata={response['messages'][-1].usage_metadata}', end='\n\n')
        for index, message in enumerate(response['messages']):
            print(f'{index+1} {type(message)}\n{message}', end='\n\n')
        print('='*100)