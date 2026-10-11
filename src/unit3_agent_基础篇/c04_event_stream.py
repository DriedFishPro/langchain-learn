import rich
from langchain.agents import create_agent
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit3_agent_基础篇.c03_agent_stream import get_weather_for_writer
from langgraph.stream import StreamTransformer, StreamChannel, ProtocolEvent


# 自定义的消息转换器，用于处理自定义的协议事件
class MyCustomTransformer(StreamTransformer):
    # 表示这个消息转换器只对"custom"模式的事件流生效，当agent运行并产生 custom 类型的事件时，就会触发该转换器的 process 方法。
    required_stream_modes = ("custom",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        # 1. 创建一个新的事件通道
        self.channel = StreamChannel()

    # 2. 把 channel 注册到 stream.extensions，通道名为custom_extension
    def init(self) -> dict[str, StreamChannel]:
        return {"custom_extension": self.channel}

    # 3. 通过channel把custom数据推送出去，外部就可以通过stream.extensions["custom_extension"]来获取了
    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "custom":
            data = event["params"]["data"]
            self.channel.push(data)
        # 返回 True 表示“放行”事件：处理完后，原始事件依然保留在主流（raw stream）中，不会被截断
        return True


model = create_model()

agent = create_agent(
    model=model,
    tools=[get_weather_for_writer],
    system_prompt="你是一个天气查询助手",
)

messages = {"messages": [{"role": "user", "content": "杭州今天天气怎么样？"}]}

# 1.原始协议事件（raw protocol events），包含完整信封（envelope）
def test_event_stream():
    stream = agent.stream_events(messages, version="v3")

    for event in stream:
        print(event, end="\n\n")

# 2.用 stream.messages 迭代消息流
def test_stream_messages():
    stream = agent.stream_events(messages, version="v3")

    for message in stream.messages:
        # 思考过程，由于langchain底层扩展字段丢失，所以这里的reasoning实际上是空的
        # print(f"{'-'*50}思考过程{'-'*50}")
        # for reasoning in message.reasoning:
        #     print(reasoning, end="", flush=True)

        # 消息chunk
        print(f'{"-" * 50} 消息chunk {"-" * 50}')
        for text in message.text:
            print(f'[{text}]', end=" ", flush=True)
        print()

        # 工具调用请求
        tool_calls = message.tool_calls.get()
        if tool_calls:
            print(f"{'-' * 50} 工具调用 {'-' * 50}")
            print(tool_calls)

        # 也可以通过message.text一次性拿完整文本
        full_text = str(message.text)
        if full_text:
            print(f"{'-' * 50} 完整消息 {'-' * 50}")
            print(full_text)


        print(f"{'-' * 50} 消息对象 {'-' * 50}")
        rich.print(message.output)

# 3.用 stream.tool_calls 迭代工具调用流
def test_stream_tool_calls():
    stream = agent.stream_events(messages, version="v3")

    for call in stream.tool_calls:
        print(f"\n工具 {call.tool_name} 开始执行，入参: {call.input}，tool_call_id：{call.tool_call_id}")

        # 注意：call.output_deltas是一个生成器，要遍历后才能获取 call.output 工具执行结果，否则output是none
        # for delta in call.output_deltas:
        #     print(delta, end="~", flush=True)

        # 或者用 list() 强制消费完生成器，驱动 LangChain 在后台完成 output 的拼接
        # list(call.output_deltas)

        print(f"\n工具执行完毕，最终结果: {call.output}", end="\n\n")

# 4.stream.values：Agent 状态快照（state snapshots），stream.output：agent 最终状态
def test_stream_values():
    stream = agent.stream_events(messages, version="v3")
    for i, value in enumerate(stream.values):
        print(f'{'-' * 50} 第{i+1}个快照 {'-' * 50}')
        rich.print(value)

    print(f"{'-' * 50} 最终状态 {'-' * 50}")
    rich.print(stream.output)

# 5.通过 stream.extensions 获取自定义扩展消息
def test_stream_extensions():
    stream = agent.stream_events(messages, version="v3", transformers=[MyCustomTransformer])
    for item in stream.extensions["custom_extension"]:
        print(f'custom_extension：{item}')

# 6.演示同时消费多种投影（适合需要同时处理文字和工具进度的场景）
def test_stream_interleave():
    stream = agent.stream_events(messages, version="v3")

    for name, item in stream.interleave("messages", "tool_calls"):
        if name == "messages":
            # item 是 message 对象
            content = str(item.text)
            print(f'content: {content}')
        elif name == "tool_calls":
            # item 是 call 对象
            print(f"\n[进度] 工具 {item.tool_name} 开始执行...")
            list(item.output_deltas)
            print(f"[完成] {item.tool_name} 执行完毕，结果：{item.output}")


if __name__ == "__main__":
    test_event_stream()
    # test_stream_messages()
    # test_stream_tool_calls()
    # test_stream_values()
    # test_stream_extensions()
    # test_stream_interleave()