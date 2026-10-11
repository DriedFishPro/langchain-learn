import os
import rich
from util.reasoning_patch import apply_patches

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from util import env_util

env_util.setup_apikey()


# 根据 provider 初始化不同平台的模型
def create_model(provider: str = "dashscope", model: str = "qwen3.7-flash-2026-07-15", temperature: float = 0.7,
                 enable_thinking: bool = False, **kwargs) -> BaseChatModel:
    apply_patches()

    if provider == "openai":
        print("正在初始化openai模型...")
        chat_model = init_chat_model(
            model=model,  # model="gpt-5.4-mini",
            model_provider=provider,  # provider="openai"
            base_url="https://api.openai-proxy.org/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,  # 控制模型输出的随机性。数值越大，输出的响应就越具有创造性；数值越小，则输出的响应就越倾向于确定性。
            max_retries=3,  # 重试次数
            timeout=120,
            reasoning={"effort": "medium", "summary": "auto"}
        )

        # from langchain_openai import ChatOpenAI
        # chat_model = ChatOpenAI(
        #     model=model,
        #     base_url="https://api.openai-proxy.org/v1",
        #     api_key=os.getenv("OPENAI_API_KEY"),
        #     temperature=temperature,
        #     timeout=120,
        # )
        return chat_model
    elif provider == "ollama":
        print("正在初始化ollama模型...")
        chat_model = init_chat_model(
            model="qwen3.5:4b",
            model_provider=provider,
            base_url="http://192.168.31.14:11434",
            temperature=temperature,
            max_retries=3,
            timeout=120,
        )

        # from langchain_ollama import ChatOllama
        # chat_model = ChatOllama(
        #     model="qwen3.5:4b",
        #     base_url="http://192.168.31.14:11434",
        #     temperature=temperature,
        #     timeout=120,
        # )
        return chat_model
    elif provider == "dashscope":
        print("正在初始化阿里云百炼模型...")

        chat_model = init_chat_model(
            model=model,
            model_provider="openai",
            base_url=f"https://{os.getenv("DASHSCOPE_WORKSPACE_ID")}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=temperature,
            max_retries=3,
            timeout=120,
            extra_body={"enable_thinking": enable_thinking},
            **kwargs
        )
        # LangChain0.3之后的版本中移除了pydantic_v1，langchain-dashscope没有升级，所以这种方式目前无法使用
        # from langchain_dashscope import ChatDashScope
        # chat_model = ChatDashScope(
        #     model=model,
        #     api_key=os.getenv("DASHSCOPE_API_KEY"),
        #     temperature=temperature,
        #     timeout=120,
        # )
        return chat_model
    else:
        raise ValueError(f"不支持的模型提供商: {provider}")


def chat_invoke(model: BaseChatModel):
    """简单对话"""
    response: AIMessage = model.invoke("你好，请你做个自我介绍")
    rich.print(response)


def chat_stream(model: BaseChatModel):
    """流式输出"""
    input = """甲、乙、丙三个人里，有一个人吃了披萨。他们每人说了一句话：
        甲说：是乙吃的。乙说：不是我吃的。丙说：也不是我吃的。
        已知：这三个人里，只有一个人说了真话。请问：披萨是谁吃的？"""

    for chunk in model.stream(input):
        print(chunk.text, end='', flush=True)


def chat_batch(model: BaseChatModel):
    """批量对话 一次性返回"""
    for response in model.batch([
        "为什么鹦鹉有五颜六色的羽毛？",
        "飞机是如何飞行的？",
    ]):
        print(response, end='\n\n')


def chat_batch_as_completed(model: BaseChatModel):
    """批量对话 逐个返回"""
    for response in model.batch_as_completed([
        "为什么鹦鹉有五颜六色的羽毛？",
        "飞机是如何飞行的？",
    ]):
        print(response, end='\n\n')


if __name__ == '__main__':
    # 通过传入不同的提供商创建不同的模型
    model = create_model()  # provider="openai", model="gpt-5.4-mini"

    chat_invoke(model)
    # chat_stream(model)
    # chat_batch(model)
    # chat_batch_as_completed(model)
    pass