from util import env_util
env_util.setup_fiddler()

from typing_extensions import TypedDict
from langchain.agents import create_agent
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from rich import print as rprint


# 1. 使用 TypedDict 定义结构
class ContactInfo(TypedDict):
    """用户联系方式"""
    name: str # 用户名
    email: str # 邮箱
    phone: str # 电话

# 2. 初始化模型
model = create_model(provider="openai", model="gpt-5.4-mini",) # 注意：结构化输出不支持thinking模式

# 3. 创建agent 并绑定结构化输出
agent = create_agent(
    model=model,
    response_format=ContactInfo,
    system_prompt="从这段文本中提取联系人信息，并以指定的 JSON 格式返回",
    # todo 避坑：
)

# 4. 运行测试
result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

rprint(result)
print(f'structured_response：{result['structured_response']}')
# {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}