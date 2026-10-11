import sqlite3
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from langgraph.store.base import IndexConfig
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.sqlite import SqliteStore

from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit2_rag_基础篇.b02_调用embedding模型 import create_embedding_model


@dataclass
class Context:
    user_id: str


def get_store():
    # sqlite连接
    conn = sqlite3.connect("sqlite/store.db", check_same_thread=False, isolation_level=None)
    # 初始化向量模型
    embeddings = create_embedding_model()
    # 使用 IndexConfig 定义嵌入模型、维度以及需要建立索引的字段，"$" 表示对整个 JSON 值进行嵌入
    index = IndexConfig(embed=embeddings, dims=1024, fields=["$"])
    # 给store配置向量模型
    store = SqliteStore(conn, index=index)
    store.setup()  # 初始化 SqliteStore，并配置 index
    return store


def test_put(store):
    namespace = ("user_preferences", "u001")
    store.put(namespace, "food", {"text": "我喜欢吃烤鸭和牛肉"})
    store.put(namespace, "job", {"text": "我是一个资深java开发者，并且也会python"})
    print('store 添加成功')


def test_search(store, query):
    namespace = ("user_preferences",)
    items = store.search(
        namespace,
        query=query, limit=1
    )
    for item in items: print(item)
    print(f'len(items)：{len(items)}')


def chat(state: MessagesState, runtime: Runtime):
    namespace = ("user_preferences", runtime.context.user_id)
    items = runtime.store.search(namespace, query=state["messages"][-1].content, limit=1)
    print(f'store检索结果: {items}')

    memories = "\n".join(item.value["text"] for item in items)
    memories = f"用户的记忆内容：\n{memories}" if memories else ""

    model = create_model()
    response = model.invoke([{"role": "system", "content": f"你是个乐于助人的助手\n{memories}"}] + state["messages"])
    return {"messages": [response]}


if __name__ == "__main__":
    store = get_store()

    workflow = StateGraph(MessagesState)
    workflow.add_node(chat)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)
    graph = workflow.compile(store=store)

    result = graph.invoke(
        input={"messages": [HumanMessage(content="我喜欢吃什么？")]},
        context=Context(user_id="u001")
    )
    print(result["messages"][-1].content)