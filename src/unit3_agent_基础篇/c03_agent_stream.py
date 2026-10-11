import time
import rich
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.prebuilt import ToolRuntime
from unit1_大模型基础篇.a03_langchain调用模型 import create_model


@tool(name_or_callable="get_weather", description="获取给定城市的天气")
def get_weather_for_writer(city: str, runtime: ToolRuntime) -> str:
    runtime.stream_writer(f"get_weather被调用：city={city}")

    for i in range(1, 10):
        runtime.stream_writer(f"{i}，")
        time.sleep(0.2)

    return f"{city} 今天（09/02）多云，28~33℃。"


# model = create_model(model="gpt-5.4-mini", provider="openai")
model = create_model()

agent = create_agent(
    model=model,
    tools=[get_weather_for_writer],
    system_prompt="你是一个天气查询助手",
)

messages = {"messages": [{"role": "user", "content": "杭州今天天气怎么样？"}]}


# 需要搭配stream_mode来使用，stream_mode的可选项有updates、messages、custom
def test_stream_messages():
    """
    当stream_mode="messages"时，表示只要LLM有响应就输出
    chunk是一个字典，包含tools和llm的消息
    """
    status = None

    for chunk in agent.stream(messages, stream_mode="messages", version="v2"):
        # rich.print(chunk)

        data = chunk["data"][0]
        if isinstance(data, AIMessageChunk):
            reasoning = data.additional_kwargs.get("reasoning_content")
            content = data.content

            # 推理过程
            if reasoning:
                if status != "reasoning":
                    print(f"\n{'=' * 50} 推理过程↓↓↓ {'=' * 50}")
                    status = "reasoning"
                print(reasoning, end="", flush=True)
            # 正文
            elif content:
                if status != "answer":
                    print(f"\n{'=' * 50} 最终回答↓↓↓ {'=' * 50}")
                    status = "answer"
                print(content, end="", flush=True)
        elif isinstance(data, ToolMessage):
            if status != "tool":
                print(f"\n{'=' * 50} 工具执行结果↓↓↓ {'=' * 50}")
                status = "tool"
            print(data.content, end='\n\n')


def test_stream_updates():
    """
    当stream_mode="updates"时，表示state中有数据更新时输出
    chunk是一个字典，包含tools和llm的消息
    """
    for chunk in agent.stream(messages, stream_mode="updates", version="v2"):
        # rich.print(chunk)
        # print('=' * 100)

        if 'tools' in chunk["data"]:
            print(f'ToolMessage：{chunk["data"]["tools"]["messages"][0].content}')
        elif 'model' in chunk["data"]:
            if chunk["data"]["model"]["messages"][0].content:
                print(f'AIMessage content：{chunk["data"]["model"]["messages"][0].content}')
            elif chunk["data"]["model"]["messages"][0].tool_calls:
                print(f'AIMessage tool_calls：{chunk["data"]["model"]["messages"][0].tool_calls}')


def test_stream_custom():
    """
    当stream_mode="messages"时，表示自定义内容输出
    chunk就是writer中的内容，v2返回的是字典：
    """
    for chunk in agent.stream(messages, stream_mode="custom", version="v2"):
        print(chunk, end="\n\n")


def test_stream_all():
    for chunk in agent.stream(messages, stream_mode=["messages", "updates", "custom"], version="v2"):
        print(chunk, end="\n\n")


if __name__ == "__main__":
    test_stream_messages()
    # test_stream_updates()
    # test_stream_custom()
    # test_stream_all()
    pass