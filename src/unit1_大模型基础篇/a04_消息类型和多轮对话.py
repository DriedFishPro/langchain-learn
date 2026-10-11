from langchain_core.messages import AIMessage, ToolMessage
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from langchain.messages import HumanMessage, SystemMessage


# 理解不同的消息类型（加入 ToolMessage）
def test_obj_messages():
    model = create_model()

    obj_messages = [
        # 设定系统人设/规则
        SystemMessage(content="你是一个实时的天气助手。"),
        # 用户发送的消息
        HumanMessage(content="北京今天天气怎么样？"),
        # 大模型返回的消息，模型发现自己无法回答，决定调用工具，当大模型决定使用工具时，它的 AIMessage 中会包含 `tool_calls` 属性
        AIMessage(
            content="",
            tool_calls=[{
                "id": "t_01",  # 关键：大模型生成的本次工具调用唯一 ID
                "name": "get_weather",  # 工具名称
                "args": {"city": "北京"},  # 工具参数
            }]
        ),
        # 外部 Python 代码执行完工具后，将结果反馈给大模型，ToolMessage 的 content 是工具执行的真实结果，且必须传入对应的 tool_call_id
        ToolMessage(
            content="晴转多云，25度",     # 调用外部工具获取的真实天气
            tool_call_id="t_01"         # 必须与上面 AIMessage 中的 id 严格对应
        ),
    ]

    ai_message = model.invoke(obj_messages)
    print(f"模型最终响应: {ai_message}")


# 不同格式的兼容性 (Object vs Dict)
def test_dict_messages():
    model = create_model()

    # 字典格式下的 ToolMessage 表达（通常需要包含 name 和 tool_call_id）
    dict_messages = [
        {"role": "system", "content": "你是一个实时的天气助手。"},
        {"role": "user", "content": "北京今天天气怎么样？"},
        # 模拟中间的工具交互 Dict
        {"role": "assistant", "content": "我正在查询北京天气...", "tool_calls": [{"id": "t_01", "name": "get_weather", "args": {"city": "北京"}}]},
        {"role": "tool", "content": "晴转多云，25度", "tool_call_id": "t_01"},
    ]

    ai_message = model.invoke(dict_messages)
    print(f"模型最终响应: {ai_message}")


# 交互式连续对话，通过 while 循环实现命令行交互式对话，直观展示每轮对话后，用户输入和AI回复都会被追加到消息列表
def test_chat():
    model = create_model()
    # 初始化消息历史（包含系统消息）
    messages = [SystemMessage(content="你是一个友好的AI助手，擅长回答各种问题。"),]

    while True:
        # 获取用户输入
        user_input = input("👤 你: ").strip()
        # 退出条件
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 再见！感谢使用~")
            break

        # 空输入跳过
        if not user_input: continue

        # 将用户消息添加到历史
        messages.append(HumanMessage(content=user_input))

        # 调用模型（此时会发送完整的消息历史）
        ai_response = model.invoke(messages)

        # 显示AI回复
        print(f"🤖 AI: {ai_response.content}\n")

        # 将AI回复添加到历史（关键步骤：这样下一轮对话时模型才能"记住"）
        messages.append(ai_response)

        # 显示当前消息历史状态
        print(f"📊 本轮对话结束，当前消息历史共 {len(messages)} 条:")
        print("-" * 60)


if __name__ == '__main__':
    # test_obj_messages()
    # test_dict_messages()
    test_chat()