import json
import sqlite3

from dataclasses import dataclass
from typing import Any, Callable
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.prebuilt import ToolRuntime
from unit1_大模型基础篇.a03_langchain调用模型 import create_model


@dataclass
class UserContext:
    user_id: str


# 1. 定义长期记忆的更新工具
def update_user_preference(type: str, content: dict[str, Any], runtime: ToolRuntime[UserContext]):
    """更新用户长期记忆

    Args:
        type: 记忆领域分类，例如：
                     - 'profile': 个人基本信息（姓名、年龄、居住地等）
                     - 'food': 饮食偏好与忌口（喜好、忌口等）
                     - 'work': 职业与技能背景
        content: 某个分类下的完整内容，比如：{"name": "张三", "job": "Java开发"}，如果content为空字典，则会删除该type
    """

    namespace = ("user_preferences", runtime.context.user_id)

    if content:
        runtime.store.put(namespace, type, content)
        print(f'长期记忆已更新：type：{type}，content：{content}', end="\n\n")
    else:
        runtime.store.delete(namespace, key=type)
        print(f'长期记忆已删除：type：{type}', end="\n\n")

    return "长期记忆已更新"


# 2. 在模型调用之前，去获取用户长期记忆，动态注入到提示词中
@wrap_model_call
def load_user_preference(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    namespace = ("user_preferences", request.runtime.context.user_id)
    store_items = request.runtime.store.search(namespace, limit=100)
    memory_data = []

    for item in store_items:
        memory_data.append({'type': item.key, 'content': item.value})

    memory_data = json.dumps(memory_data, indent=4, ensure_ascii=False) if memory_data else "没有找到用户的长期记忆数据"
    system_prompt = request.system_message.content.replace('{memory_data}', memory_data)

    print(f'加载用户长期记忆：{memory_data}', end="\n\n")
    return handler(request.override(system_message=SystemMessage(content=system_prompt)))


if __name__ == "__main__":
    # 短期记忆
    conn_checkpointer = sqlite3.connect("./sqlite/agent_memory.db", check_same_thread=False)  # check_same_thread=False 允许跨线程复用连接
    checkpointer = SqliteSaver(conn_checkpointer)
    checkpointer.setup()  # 初始化数据库

    # 长期记忆 (持久化偏好，跨 Thread 生效)
    conn_store = sqlite3.connect("./sqlite/agent_store.db", check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn_store)
    store.setup()  # 初始化数据库

    # 组装 Agent (注入 store)
    system_prompt = """你是一个具备长期记忆管理能力的智能助手。请根据与用户的对话，实时维护并更新用户的长期记忆数据。

    【用户当前长期记忆】
    <user_memory_data>
    {memory_data}
    </user_memory_data>

    【记忆管理与更新机制】
    你只需通过调用 `update_user_preference(type: str, content: dict)` 工具来维护记忆。
    注意：该工具采用【全量更新机制】，即你传入的 `content` 字典会完全覆盖该 `type` 原有的字典内容。

    请严格遵循以下变更逻辑：
    1. 分类管理 (type)
       根据记忆主题选择或新建合理的分类 Key（如：`profile` 个人基本信息、`work` 职业技能、`food` 饮食偏好、`hobby` 兴趣爱好等）。
    2. 内容更新规则 (content)
       对照 `<user_memory_data>` 中当前分类的已有数据，按以下三种情形组装最新的 `content` 字典：
       - 【情形一：新增/扩展属性】
         * 分类不存在：新建分类 `type`，并将提取到的属性写入 `content` 字典。
         * 分类已存在：在原有 `content` 字典的基础上追加新的 key-value 属性，提交包含新旧所有属性的完整字典。
       - 【情形二：修改现有属性（语义判定）】
         * 覆盖模式 (Overwrite)：若用户在纠正输入错误（如年龄从 300 岁修正为 30 岁），或更新具有排他性的单值属性（如手机号、居住地变更），直接使用最新值替换旧值。
         * 合并模式 (Semantic Merge)：若信息具备累加性（如掌握的技能、偏好列表）或存在历史演进关系（如转行/跳槽），
           应将新旧语义进行整合（例如：技能由 ["Java"] 更新为 ["Java", "Python"]；职业由 "开发者" 更新为 "曾经是开发者，现为外卖员"）。
       - 【情形三：删除/清空记忆】
         * 属性级删除：若用户要求忘记分类中的某项特定信息（如“忘了我的年龄”），从原有 `content` 中剔除该 key，并将剩余属性构成的字典进行全量提交。
         * 分类级清空：若用户要求清空整个分类（或删除后该分  类下已无任何属性），必须传入空字典 `content: {}`，系统将自动注销该分类。
    3. 注意事项：每次更新某一分类时，严禁遗漏该分类下未被修改的其他已有 key。
    """

    agent = create_agent(
        model=create_model(enable_thinking=True),
        tools=[update_user_preference],
        middleware=[load_user_preference],
        checkpointer=checkpointer,
        store=store,  # <--- 将长期记忆store注入给 Agent
        context_schema=UserContext,
        system_prompt=system_prompt,
    )

    # 切换 thread_id 测试，你会发现它忘记了聊天记录，但依然能记得用户
    config = {"configurable": {"thread_id": 'user_03_03'}}
    print("输入 'quit' 或 'exit' 退出对话\n")

    while True:
        user_input = input("你: ").strip()

        response = agent.invoke(
            input={"messages": [{"role": "user", "content": user_input}]},
            config=config,
            context=UserContext(user_id="user_03"),
        )

        print(response['messages'][-1].content)
        print("=" * 100)