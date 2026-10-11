import json
from pathlib import Path
from typing import Annotated, List, Literal, Dict
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from qdrant_client.http import models
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit4_智能保险客服助手_v1.data_process.init_qdrant import init_qdrant_store


class GraderSchema(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="评估结果：yes-评估通过，no-评估不通过")
    feedback: str = Field(default="", description="评估反馈：简要说明不通过的原因，并给出下一轮检索的明确建议")


class RagState(TypedDict):
    messages: Annotated[list, add_messages]  # 完整的对话历史，用于消指代和上下文理解
    source: str  # 目标保险产品全称（主网关传过来）
    grade: GraderSchema  # 质检评级
    loop_count: int  # 重试计数器（卡死保护）


class RagGraph:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.collection = "unit4_rag"
        self.qdrant_store = init_qdrant_store(self.collection, str(self.base_dir / "db" / "qdrant"))

        self.tools = [
            tool(self.get_catalogue),
            tool(self.similarity_search, parse_docstring=True),
            tool(self.metadata_search, parse_docstring=True)
        ]

        # 可以根据任务分配不同的模型
        self.intention_model = create_model().bind_tools(self.tools)  # 意图分析模型
        self.evaluate_model = create_model().with_structured_output(GraderSchema)  # 结果评估模型
        self.generate_model = create_model(temperature=0)  # 最终回答模型

    def similarity_search(self, query: str, config: Annotated[RunnableConfig, InjectedToolArg()]):
        """
        在向量数据库中进行相似度搜索。

        Args:
            query: 根据用户问题或正文线索动态构造的检索语句
        """

        # 从自动注入的 config 中安全获取运行时参数
        source = config["configurable"].get("source", None)
        print(f"[rag graph] 相似度检索 | query: {query} | source: {source}")

        # 构建 Qdrant 的条件过滤器
        qdrant_filter = models.Filter(
            must=[models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))]
        )

        results = self.qdrant_store.similarity_search_with_score(query=query, k=5, filter=qdrant_filter)
        page_content_list = ['未找到相关知识库内容']

        for doc, score in results:
            print(f"    ↳ [已捞取] score={score:.4f}] doc.metadata={doc.metadata}")
            # print(doc.page_content, end="\n\n")

        page_content_list = [doc.page_content for doc, score in results] if results else page_content_list
        result = json.dumps(page_content_list, ensure_ascii=False)
        print(f'[rag graph] 相似度检索到{len(page_content_list)}个文档', end="\n\n")
        return result

    def metadata_search(self, header_level: str, header_value: str,
                        config: Annotated[RunnableConfig, InjectedToolArg()]):
        """
        在向量数据库中进行元数据搜索。

        Args:
            header_level: 从面包屑导航中动态判断层级，第一层用"header1"，第二层用"header2"，第三层用"header3"
            header_value: 从面包屑导航中动态提取对应层级的标题文本，必须来自实际检索结果，不可自行编造
        """

        # 1. 从自动注入的 config 中安全获取运行时参数,确保只在当前保险条款中检索
        source = config["configurable"].get("source", None)
        print(
            f"[rag graph] 元数据检索 | header_level: {header_level} | header_value: {header_value} | source: {source}")

        # 2. 构建组合过滤条件
        must_conditions = [
            models.FieldCondition(key=f"metadata.{header_level}", match=models.MatchValue(value=header_value)),
            models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))
        ]

        qdrant_filter = models.Filter(must=must_conditions)

        # 3. 使用底层 client 的 scroll 方法,直接按 Filter 获取记录(0 向量参与,性能极高)
        records, next_page_offset = self.qdrant_store.client.scroll(
            collection_name=self.collection,
            scroll_filter=qdrant_filter,
            limit=100,  # 一次性捞取该大章下的前 100 个分片(通常足够容纳一整章)
            with_payload=True,  # 必须为 True,否则拿不到文本
            with_vectors=False  # 不需要返回向量值
        )

        # 4. 提取文本(LangChain 存入 Qdrant 时,文本正文默认放在 payload 的 "page_content" 字段中)
        page_content_list = []

        for record in records:
            payload = record.payload
            if payload and "page_content" in payload:
                # 打印调试信息,看一眼拿到了哪些片段
                print(f"    ↳ [已捞取] metadata: {payload.get('metadata')}")
                # print(f"        page_content: {payload['page_content']}", end="\n\n")
                page_content_list.append(payload["page_content"])

        # 5. 序列化并返回
        result = json.dumps(page_content_list or ['未找到相关知识库内容'], ensure_ascii=False)
        print(f'[rag graph] 元数据检索到{len(page_content_list)}个文档', end="\n\n")
        return result

    def get_catalogue(self, config: Annotated[RunnableConfig, InjectedToolArg()]):
        """获取文档目录"""
        source = config["configurable"].get("source", None)
        print(f"[rag graph] 获取文档目录，source={source}")
        must_conditions = [models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))]
        qdrant_filter = models.Filter(must=must_conditions)

        # 3. 使用底层 client 的 scroll 方法,直接按 Filter 获取记录(0 向量参与,性能极高)
        records, next_page_offset = self.qdrant_store.client.scroll(
            collection_name=self.collection,
            scroll_filter=qdrant_filter,
            limit=200,
            with_payload=True,  # 必须为 True,否则拿不到文本
            with_vectors=False  # 不需要返回向量值
        )

        print(f'检索到{len(records)}个文档目录')
        metadatas = [record.payload.get("metadata") for record in records]
        metadatas.sort(key=lambda x: x.get("index", 0))
        for m in metadatas: m.pop("source")  # 删除元数据中的 source 字段，减少数据量

        print(metadatas)
        return json.dumps([json.dumps(m, ensure_ascii=False) for m in metadatas], ensure_ascii=False)

    def intention_analysis_node(self, state: RagState, config: RunnableConfig):
        source = config["configurable"].get("source", None)
        feedback = getattr(state.get("grade"), "feedback", None) or "无"

        prompt_template = (self.base_dir / "prompt" / "rag_intent_prompt.md").read_text(encoding="utf-8")
        prompt = prompt_template.replace('{source}', source).replace('{feedback}', feedback)
        print(f"[rag_intent_analyze_node] 意图分析开始 input={state['messages'][0].content}")

        messages = [SystemMessage(content=prompt)] + state['messages']
        ai_message = self.intention_model.invoke(messages, config)

        print(f"[rag_intent_analyze_node] 意图分析结果: {ai_message}", end="\n\n")
        return {"messages": [ai_message], "loop_count": state.get("loop_count", 0) + 1}

    def retriever_node(self, state: RagState, config: RunnableConfig):
        print("[retriever_node] 检索节点开始执行工具...")

        # 实例化 LangGraph 预置的 ToolNode 执行器
        tool_node = ToolNode(self.tools)
        # 显式执行 tool_executor 并返回状态更新结果
        result = tool_node.invoke(state, config)

        print(f"[retriever_node] 检索完毕：result={result}", end='\n\n')
        return result

    def evaluate_node(self, state: RagState):
        print(f"[evaluate_node] evaluate 评估开始")
        prompt = (self.base_dir / "prompt" / "rag_evaluate_prompt.md").read_text(encoding="utf-8")

        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = self.evaluate_model.invoke(messages)

        print(f"[evaluate_node] evaluate 评估结果: {response}", end="\n\n")
        return {'grade': response}

    def generate_answer_node(self, state: RagState, config: RunnableConfig):
        doc_set = set()

        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                doc_set.update(json.loads(msg.content))

        doc_list = list(doc_set)
        print(f"[rag_generate_answer_node] 生成回复 docs: {len(doc_list)}")

        source = config["configurable"].get("source", None)
        print(f"[generate_answer_node] 生成回复 source: {source}", end="\n\n")

        docs = json.dumps(doc_list, ensure_ascii=False) if len(doc_list) > 0 else '未检索到相关文档'
        question = state["messages"][0].content

        prompt_template = (self.base_dir / "prompt" / "rag_generate_prompt.md").read_text(encoding="utf-8")
        prompt = prompt_template.replace('{source}', source).replace('{docs}', docs).replace('{question}', question)

        response = self.generate_model.invoke([SystemMessage(content=prompt)])
        print(f"[rag_generate_answer_node] 生成回复: {response}", end="\n\n")
        return {"messages": [response], "loop_count": 0}

    def route_after_intention(self, state: RagState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "retriever"
        return "generate_answer"

    def route_after_evaluate(self, state: RagState):
        if state["loop_count"] >= 3:
            print('超出重试上限')
            return "generate_answer"

        if state["grade"].binary_score == "yes":
            print('评估通过，开始生成结果')
            return "generate_answer"
        else:
            print(f'评估不通过，开始次{state["loop_count"]}次重试')
            return "intention_analysis"

    def build(self):
        builder = StateGraph(RagState)
        builder.add_node("intention_analysis", self.intention_analysis_node)
        builder.add_node("retriever", self.retriever_node)
        builder.add_node("evaluate", self.evaluate_node)
        builder.add_node("generate_answer", self.generate_answer_node)

        builder.add_edge(START, "intention_analysis")
        builder.add_conditional_edges("intention_analysis", self.route_after_intention, {
            "retriever": "retriever",
            "generate_answer": "generate_answer"
        })
        builder.add_edge("retriever", "evaluate")
        builder.add_conditional_edges("evaluate", self.route_after_evaluate, {
            "intention_analysis": "intention_analysis",
            "generate_answer": "generate_answer"
        })
        builder.add_edge("generate_answer", END)
        return builder.compile()


if __name__ == "__main__":
    rag_graph = RagGraph()
    graph = rag_graph.build()
    # draw_graph(graph, "./rag_graph.png")

    while True:
        query = input('请输入想要检索的内容：')
        if not query: continue

        result = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            {"configurable": {"source": "长生爱康保重大疾病保险"}}
        )

        print(f"最终回答：{result['messages'][-1].content}")
        print('=' * 100)

    rag_graph.qdrant_store.client.close()