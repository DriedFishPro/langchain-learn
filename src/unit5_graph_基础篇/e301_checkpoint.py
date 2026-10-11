import os
import sqlite3
from typing import Annotated

import rich
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from typing_extensions import TypedDict
from uuid_utils import uuid4


class State(TypedDict):
    topic: str
    messages: Annotated[list[AnyMessage], add_messages]


def generate_topic(state: State):
    print(f"generate_topic state={state}", end="\n\n")
    human_message = HumanMessage(content=f"讲一个关于{state['topic']}的笑话")
    return {"messages": [human_message]}


def write_joke(state: State):
    print(f"write_joke state={state}", end="\n\n")
    ai_message = AIMessage(content=f'假装这是ai回复的关于【{state['topic']}】的内容-{str(uuid4())}')
    return {"messages": [ai_message]}

# 打印历史记录
def print_history(graph: CompiledStateGraph, config):
    history = list(graph.get_state_history(config))
    print(f"{'='*50} history ↓↓↓ {'='*50}")
    for i, state in enumerate(history, start=1):
        print(i)
        rich.print(state)
        print('-' * 100)
    print(f"{'=' * 50} history ↑↑↑ {'=' * 50}")

# replay：重新执行write_joke节点
def test_replay(graph: CompiledStateGraph, config):
    print(f'\n{'=' * 50} test_replay start {'=' * 50}')
    state_history = list(graph.get_state_history(config))
    before_joke_state = next(s for s in state_history if s.next == ("write_joke",))

    print(f'before_joke_state：{before_joke_state}', end="\n\n")
    result = graph.invoke(input=None, config=before_joke_state.config)

    print(f'test_replay result：{result}')
    print(f'{'=' * 50} test_replay end {'=' * 50}', end="\n\n")

# fork：先修改topic，再重新执行write_joke节点
def test_fork(graph: CompiledStateGraph, config):
    print(f"test_fork start {'=' * 100}")
    history = list(graph.get_state_history(config))
    before_joke = next(s for s in history if s.next == ("write_joke",))

    fork_config = graph.update_state(
        before_joke.config,
        values={"topic": "狗"},
        # as_node：从哪个节点的视角写入状态，假装这是 generate_topic 产生的更新，所以会从 write_joke 开始继续
        as_node="generate_topic",
    )

    rich.print(fork_config)
    graph.invoke(None, fork_config)
    print(f'test_fork end {'=' * 100}')


if __name__ == "__main__":
    workflow = StateGraph(State)
    workflow.add_node(generate_topic)
    workflow.add_node(write_joke)
    workflow.add_edge(START, "generate_topic")
    workflow.add_edge("generate_topic", "write_joke")
    workflow.add_edge("write_joke", END)

    os.makedirs('sqlite', exist_ok=True)
    conn = sqlite3.connect("sqlite/checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    graph = workflow.compile(checkpointer=checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": '004'}}

    result = graph.invoke({"topic": "猫"}, config)
    rich.print(result)

    test_replay(graph, config)
    print_history(graph, config)

    # test_fork(graph, config)
    # print_history(graph, config)