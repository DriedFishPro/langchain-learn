from util.env_util import setup_langsmith
setup_langsmith()

import json
from pathlib import Path

from pydantic import BaseModel, Field
from typing import Annotated, TypedDict, List, Literal, Optional
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit6_智能保险客服助手_v2.graph.rag_graph import RagGraph
from unit6_智能保险客服助手_v2.graph.sql_graph import SqlGraph


# 意图分析结构化输出
class IntentSchema(TypedDict):
    intent: Annotated[
        Literal["chitchat", "knowledge", "data", "joint"],
        "意图分类：chitchat(闲聊寒暄), data(业务数据查询), knowledge(知识查询), joint(组合查询:业务数据+产品内容)"
    ]
    product_names: Annotated[
        Optional[List[str]],
        "对齐后的保险产品全称"
    ]
    sql_query_rewrite: Annotated[
        Optional[str],
        "改写后的业务数据检索诉求，仅在 intent 为 joint 时提供，否则为 null"
    ]
    rag_query_rewrite: Annotated[
        Optional[List[str]],
        "改写后的保险知识检索诉求列表。多产品比对时每维度一项；joint 组合查询时为单元素列表；其余场景为 null"
    ]

# 状态定义
class MasterState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: IntentSchema            # 意图分类
    sql_result: Optional[str]       # SQL 查询结果
    rag_result: Optional[str]       # RAG 检索结果

class MasterGraph:
    def __init__(self):
        print('[MasterGraph] 实例化')
        self.base_dir = Path(__file__).resolve().parent.parent
        self.intention_model = (create_model(response_format={"type": "json_object"})
                                .with_structured_output(IntentSchema))
        self.generate_model = create_model(temperature=0)

        # 实例化两个子graph
        self.sql_graph_builder = SqlGraph()
        self.sql_graph = self.sql_graph_builder.build()
        self.rag_graph_builder = RagGraph()
        self.rag_graph = self.rag_graph_builder.build()

        # 产品字典
        res = self.sql_graph_builder.execute_sql_node({"messages": [HumanMessage(content="SELECT * FROM products")]})
        self.product_dict: List[dict[str, str]] = json.loads(res["messages"][-1].content)


    # 节点1：意图分析
    def intention_analysis_node(self, state: MasterState):
        print("[intention_analysis_node] 开始意图分析")

        prompt_template = (self.base_dir / "prompt" / "master_intent_prompt.md").read_text(encoding="utf-8")
        prompt = prompt_template.replace('{products}', json.dumps(self.product_dict, ensure_ascii=False))
        # print(f"[intention_analysis_node] 意图分析prompt: {prompt}", end="\n\n")

        messages = [SystemMessage(content=prompt)] + state["messages"]
        intent = self.intention_model.invoke(messages)

        print(f"[intention_analysis_node] 意图分析结果: {intent}", end="\n\n")
        return {"intent": intent}

    def sql_graph_node(self, state: MasterState):
        print("[sql_graph_node] 调用 SQL Graph...")
        # 如果是 joint 意图，优先使用 LLM 改写过的 query
        sql_query = state['intent']['sql_query_rewrite'] or state['messages'][-1].content
        result = self.sql_graph.invoke({"messages": [HumanMessage(content=sql_query)]})

        sql_res_str = result['messages'][-1].content  # sql_graph返回的原始结果字符串
        product_name_list = state['intent']['product_names']

        # 仅在联合查询时，尝试通过“确定性属性判断”，从SQL结果中动态剥离出产品名称
        if state['intent']['intent'] == "joint":
            try:
                db_rows = json.loads(sql_res_str)
                first_row = db_rows[0]

                # 确定性检查：判断返回的数据行中是否包含产品名称的属性标签
                if "product_name" in first_row:
                    product_name_list = [row.get("product_name") for row in db_rows]
                    print(f"[sql_graph_node] 成功解析出关联产品: {product_name_list}")
            except Exception as e:
                print(f"[sql_graph_node] 尝试解析 SQL 结果中的产品失败: {e}")

        # 更新意图分类中的 product_names
        intent = {**state['intent'], "product_names": product_name_list}
        return {"sql_result": sql_res_str, "intent": intent}

    def rag_graph_node(self, state: MasterState):
        print("[rag_graph_node] 调用 RAG Graph...")
        rag_query_list = state['intent']['rag_query_rewrite'] or [state['messages'][-1].content]
        rag_results = []

        for rag_query in rag_query_list:
            for product in state['intent']['product_names']:
                result = self.rag_graph.invoke(
                    {"messages": [HumanMessage(content=rag_query)]},
                    {"configurable": {"source": product}}
                )
                rag_results.append({"query": rag_query, "product": product, "content": result['messages'][-1].content})
        return {"rag_result": json.dumps(rag_results, ensure_ascii=False)}

    def generate_answer_node(self, state: MasterState):
        print("[generate_answer_node] 开始生成回复...")
        prompt_template = (self.base_dir / "prompt" / "master_generate_prompt.md").read_text(encoding="utf-8")

        prompt = (
            prompt_template
            .replace('{products}', json.dumps(self.product_dict, ensure_ascii=False))
            .replace('{intent}', state["intent"]['intent'])
            .replace('{sql_result}', state.get("sql_result") or "未检索到数据")
            .replace('{rag_result}', state.get("rag_result") or "未检索到内容")
        )

        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = self.generate_model.invoke(messages)
        print(f"[generate_answer_node] 生成回复完成: {response.content}")

        # 4. 返回最终答案，顺便清空本次的中间件数据（防止污染下一轮对话）
        return {"messages": [response], "sql_result": None, "rag_result": None}

    def route_after_intent(self, state: MasterState) -> Literal["sql_graph", "rag_graph", "generate_answer"]:
        route_map = {"chitchat": "generate_answer", "knowledge": "rag_graph", "joint": "sql_graph",
                     "data": "sql_graph", }
        intent = state["intent"]['intent']
        return route_map.get(intent, "generate_answer")

    def route_after_sql(self, state: MasterState) -> Literal["rag_graph", "generate_answer"]:
        intent = state["intent"]['intent']
        return "rag_graph" if intent == "joint" else "generate_answer"

    def build_graph(self):
        # 4. 构建确定性工作流
        print('[MasterGraph] build_graph')
        workflow = StateGraph(MasterState)
        workflow.add_node("intention_analysis", self.intention_analysis_node)
        workflow.add_node("sql_graph", self.sql_graph_node)
        workflow.add_node("rag_graph", self.rag_graph_node)
        workflow.add_node("generate_answer", self.generate_answer_node)

        # 串联执行
        workflow.add_edge(START, "intention_analysis")
        workflow.add_conditional_edges("intention_analysis", self.route_after_intent)
        workflow.add_conditional_edges("sql_graph", self.route_after_sql)
        workflow.add_edge("rag_graph", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile(checkpointer=InMemorySaver())  #

if __name__ == "__main__":
    master_graph = MasterGraph()
    graph = master_graph.build_graph()
    # draw_graph(graph, "../image/master_graph.png")

    print('智能保险客服问答助手已启动，请输入你想咨询的问题')
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("你：").strip()
        if not user_input: continue

        result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        # rich.print(result)
        print('=' * 100, end='\n\n')

    # 客户张伟购买的是哪个保险产品？他这款保险的免责条款有哪些？
    master_graph.rag_graph_builder.qdrant_store.client.close()