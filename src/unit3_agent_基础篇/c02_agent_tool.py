import rich
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langgraph.prebuilt import ToolRuntime
from unit1_大模型基础篇.a03_langchain调用模型 import create_model


class Location(BaseModel):
    city: str = Field(description="城市名称，例如：杭州市")
    district: Optional[str] = Field(None, description="区/县，例如：西湖区")


class WeatherInput(BaseModel):
    location: Location = Field(description="地理位置详细信息")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius", description="温度单位偏好，中国地区用摄氏度，美国地区用华氏度")


@tool(name_or_callable='get_location', description='获取用户所在地区')
def get_location_mock()-> str:
    print("get_location 被调用")
    # return "纽约市皇后区"
    return "杭州市西湖区"


@tool(args_schema=WeatherInput, description='获取某个地点的天气信息', return_direct=True)
def get_weather1(location: Location, units: str) -> str:
    print(f"get_weather被调用：location={location}, units={units}")
    district = location.district if location.district else ''
    return f"{location.city} {district} 今天（08/02）多云，温度28~33"


@tool(name_or_callable="get_weather", parse_docstring=True)
def get_weather2(city: str, units: str="celsius") -> str:
    """获取给定城市的天气

    Args:
        city: 城市名称
        units: 温度单位偏好，中国地区用摄氏度，美国地区用华氏度
    """
    print(f"get_weather被调用：city={city}, units={units}")
    return f"{city} 今天（08/02）多云，温度28~33"


class UserContext(BaseModel):
    user_id: str = Field(description="用户id")


def get_weather3(city: str, runtime: ToolRuntime) -> str:
    """获取给定城市的天气

    Args:
        city: 城市名称
    """
    state = runtime.state
    print(f'state：{state}', end="\n\n")

    config = runtime.config
    print(f'config：{config}', end="\n\n")

    context = runtime.context
    print(f'context：{context}', end="\n\n")
    return f"今天（08/02）多云，温度28~33"


if __name__ == '__main__':
    tool_schema = convert_to_openai_tool(get_weather3)
    rich.print(tool_schema)

    model = create_model()
    agent = create_agent(
        model=model,
        tools=[get_location_mock, get_weather3],
        system_prompt="你是一个天气查询助手",
        checkpointer=InMemorySaver(),
        context_schema=UserContext
    )

    config = {'configurable': {'thread_id': '001'}}
    context = UserContext(user_id='9527')
    messages = {"messages": [{"role": "user", "content": "今天天气怎么样？"}]}

    result = agent.invoke(input=messages, config=config, context=context)
    print(result['messages'][-1])