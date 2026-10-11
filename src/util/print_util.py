import rich

def print_event_stream(stream):
    for name, item in stream.interleave("messages", "tool_calls"):
        if name == "messages":
            # 1. 先遍历 item.text 流式打印ai回复内容
            has_text = False
            for delta in item.text:
                if not has_text: print(f'{"-" * 50} 模型回复 {"-" * 50}')
                print(delta, end="", flush=True)
                has_text = True  # 只要输出了字符，就标记为已拿到文本

            # 2. 如果消息中没有文本输出，说明这次消息是【工具调用请求】
            tool_calls = item.tool_calls.get()
            if tool_calls:
                print(f'{"-" * 50} 工具调用 {"-" * 50}')
                rich.print(tool_calls)
                print()
        elif name == "tool_calls":
            # 3. 工具执行结果
            list(item.output_deltas)
            print(f'{"-" * 50} {item.tool_name} 执行结果 {"-" * 50}')
            print(item.output.content)
            print()

    print('\n\n' + '='*100 + '\n')