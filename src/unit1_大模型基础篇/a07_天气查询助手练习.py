import os
import requests
import rich
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from unit1_大模型基础篇.a03_langchain调用模型 import create_model


def get_adcode(city: str):
    """根据城市名称获取对应的行政区划代码

    Args:
        city: 城市名称（不能有省市区这些行政区划后缀，如：杭州市->杭州，上海市->上海）
    """
    adcode_map = {
        "北京": "110000",
        "上海": "310000",
        "广州": "440100",
        "深圳": "440300",
        "杭州": "330100",
        "南京": "320100",
        "成都": "510100",
        "武汉": "420100",
        "西安": "610100",
        "重庆": "500000"
    }

    adcode = adcode_map.get(city)
    print(f'获取行政区划工具被调用，city={city}，adcode={adcode}', end='\n\n')

    if adcode is None:
        return f"暂不支持该城市：{city}"

    return adcode


def get_weather(adcode: str) -> dict:
    """获取指定地区的天气信息

    Args:
        adcode: 行政区划代码
    """
    try:
        # 高德地图天气API配置
        api_key = os.getenv("GAO_DE_API_KEY")
        url = "https://restapi.amap.com/v3/weather/weatherInfo"

        # 构建请求参数 base返回实况天气，all返回预报天气
        params = {"key": api_key, "city": adcode, "extensions": "base"}
        response = requests.get(url, params=params, timeout=10)

        print(f"天气查询工具被调用")
        rich.print(response.json())
        return response.json()
    except Exception as e:
        return f"天气查询出错：{str(e)}"


def weather_bot():
    model = create_model(enable_thinking=True)
    model_with_tools = model.bind_tools([get_adcode, get_weather])
    tools_map = {"get_adcode": get_adcode, "get_weather": get_weather}

    # 初始化消息历史（包含系统消息）
    messages = [SystemMessage(content="你是一个天气查询助手。")]
    print("天气查询助手已启动，请输入你的问题")

    # 外层这个是多轮对话的循环
    while True:
        user_input = input("你: ").strip()
        messages.append(HumanMessage(content=user_input))

        # 内层循环是在一轮对话中多次调用工具的
        while True:
            ai_message = model_with_tools.invoke(messages)
            messages.append(ai_message)

            # 如果大模型返回了tool_calls，则表示需要调用本地工具
            if ai_message.tool_calls:
                for tool_call in ai_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]   # 必须拿到 id，大模型靠这个 id 来匹配是谁（哪条消息）的执行结果

                    # 1.找到本地函数并执行
                    target_tool = tools_map.get(tool_name)
                    tool_output = target_tool(**tool_args)

                    # 2.将工具调用结果封装成ToolMessage并追加到消息列表中
                    tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_id)
                    messages.append(tool_message)
            else:
                print("没有工具调用，流程结束")
                break

        for i, message in enumerate(messages):
            print(f"消息列表-{i+1} {type(message)}：\n {message}", end="\n\n")


if __name__ == '__main__':
    # get_weather(get_adcode("杭州"))
    weather_bot()