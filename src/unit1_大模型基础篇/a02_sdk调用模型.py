import os
from util import env_util
env_util.setup_apikey()


# 使用openai sdk（兼容）调用阿里云百炼平台的模型
def chat_openai(messages):
    from openai import OpenAI
    # 文档链接：https://bailian.console.aliyun.com/cn-beijing?spm=5176.29619931.J_gskoo9IcCWEOwPmyvhNFm.1.74cd10d7EIUr2K&tab=doc#/doc/?type=model&url=2841718
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=f"https://{os.getenv("DASHSCOPE_WORKSPACE_ID")}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen3.6-flash",
        messages=messages,
    )

    print(completion.model_dump_json())
    # print(completion.choices[0].message.content)


# 使用ollama sdk调用本地ollama中的模型
def chat_ollama(messages):
    import ollama
    client = ollama.Client(host='http://192.168.31.14:11434')

    response = client.chat(
        model='qwen3.5:4b',
        messages=messages
    )
    print(response.model_dump_json())


# 使用dashscope sdk调用阿里云百炼平台的模型
def chat_dashscope(messages):
    import dashscope
    # 文档链接：https://bailian.console.aliyun.com/cn-beijing?spm=5176.29619931.J_gskoo9IcCWEOwPmyvhNFm.1.74cd10d7EIUr2K&tab=doc#/doc/?type=model&url=2841718
    # 除Qwen3.7-Max系列和qwen3.6-max-preview外，Qwen3.6、Qwen3.5系列需要使用多模态接口，直接替换模型会导致报错
    response = dashscope.MultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen3.6-flash",
        messages=messages,
        result_format="message",
    )

    print(response)


if __name__ == '__main__':
    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手"},
        {"role": "user", "content": "你好，请你做个自我介绍。"},
    ]

    # chat_ollama(messages)
    chat_openai(messages)
    # chat_dashscope(messages)