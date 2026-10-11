from util import env_util
# env_util.setup_fiddler()

from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from unit1_大模型基础篇.a03_langchain调用模型 import create_model


# 1. 使用 TypedDict 定义结构
class ContactInfo(TypedDict):
    """用户联系方式"""
    name: str # 用户名
    email: str # 邮箱
    phone: str # 电话

# 2. 初始化模型，不同厂商的模型对于结构化输出的支持不太一样
model = create_model(enable_thinking=False, response_format = {"type": "json_object"}) # 百炼平台的千问系列模型需要在提示词中包含 JSON 关键词
# model = create_model(model="gpt-5.4-mini", provider="openai")
print(f'type model: {type(model)}')

# 3. 绑定结构化输出
structured_model = model.with_structured_output(ContactInfo)
print(f'type structured_model: {type(structured_model)}')


# 使用普通字符串模板
def test_structured_output1():
    # 4. 组装并调用
    prompt_template = "从下面这段文本中提取联系人信息，并以 JSON 格式返回。文本内容：{text}"
    input = prompt_template.format(text="我叫张三，我的联系电话是13288888888，我的邮箱是88888888@gmail.com")

    result = structured_model.invoke(input)
    print(type(result), result)
    # {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}

# 使用提示词模板
def test_structured_output2():
    prompt_template = ChatPromptTemplate.from_template("从这段文本中提取联系人信息，并以 JSON 格式返回: {text}")
    prompt_value = prompt_template.invoke({"text": "我叫张三，我的联系电话是13288888888，我的邮箱是88888888@gmail.com"})
    print(prompt_value)
    result = structured_model.invoke(prompt_value)
    print(result)

# 使用管道符来实现链式调用
def test_structured_output3():
    prompt_template = ChatPromptTemplate.from_template("从这段文本中提取联系人信息，并以 JSON 格式返回: {text}")
    chain = prompt_template | structured_model
    result = chain.invoke({"text": "我叫张三，我的联系电话是13288888888，我的邮箱是88888888@gmail.com"})
    print(result)


if __name__ == '__main__':
    test_structured_output1()
    # test_structured_output2()
    # test_structured_output3()