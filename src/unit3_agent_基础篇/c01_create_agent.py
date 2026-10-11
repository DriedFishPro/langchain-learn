import rich
from langchain.agents import create_agent
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit1_大模型基础篇.a07_天气查询助手练习 import get_adcode, get_weather

model = create_model()
agent = create_agent(
    model=model,
    tools=[get_adcode, get_weather],
    system_prompt="你是一个天气预报员",
)


messages = {"messages": [{"role": "user", "content": "杭州今天天气怎么样？"}]}
result = agent.invoke(input=messages)
rich.print(result)