import rich
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from unit1_大模型基础篇.a03_langchain调用模型 import create_model

from util.env_util import setup_fiddler
setup_fiddler()


def get_weather(city: str) -> str:
    """获取某个地点的天气信息。

    Args:
        city: 城市名称，例如 "北京"、"上海"
    """
    return f"{city} 今天多云，20~28℃。"


def test_bind_tools():
    """绑定工具"""
    model = create_model(enable_thinking=True)
    model_with_tools = model.bind_tools([get_weather])

    messages = [HumanMessage(content="杭州天气怎么样？")]
    ai_message = model_with_tools.invoke(messages)
    rich.print(ai_message, end="\n\n")


if __name__ == '__main__':
    test_bind_tools()

    # 查看 tool_schema
    tool_schema = convert_to_openai_tool(get_weather)
    rich.print(tool_schema)