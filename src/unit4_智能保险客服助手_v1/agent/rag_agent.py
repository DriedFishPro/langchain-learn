import json
from dataclasses import dataclass
from typing import Callable
from pathlib import Path

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolRuntime
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.tools import tool
from qdrant_client.http import models

from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit4_智能保险客服助手_v1.data_process.init_qdrant import init_qdrant_store


@dataclass
class RagContext:
    source: str


class RagAgent:
    def __init__(self):
        # 1. 资源初始化(只在实例化时执行一次)
        self.collection = "unit4_rag"
        self.base_dir = Path(__file__).resolve().parent.parent
        self.qdrant_store = init_qdrant_store(self.collection, str(self.base_dir / "db" / "qdrant"))
        self.model = create_model(temperature=0)
        system_prompt = (self.base_dir / "prompt" / "rag_prompt.md").read_text(encoding="utf-8")

        self.agent = create_agent(
            model=self.model,
            system_prompt=system_prompt,
            tools=[
                tool(self.similarity_search, parse_docstring=True),
                tool(self.metadata_search, parse_docstring=True),
                tool(self.get_catalogue)
            ],
            context_schema=RagContext,
            middleware=[self.add_context]
        )

    @staticmethod
    @wrap_model_call
    def add_context(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        source = request.runtime.context.source
        prompt = request.system_message.content.replace('{source}', source)
        system_message = SystemMessage(content=prompt)
        return handler(request.override(system_message=system_message))

    def similarity_search(self, query: str, runtime: ToolRuntime[RagContext]):
        """
        知识库相似度搜索

        Args:
            query: 根据用户问题或正文线索动态构造的检索语句
        """

        source = runtime.context.source
        print(f"[rag agent] 相似度检索 | query: {query} | source: {source}")

        # 构建 Qdrant 的条件过滤器
        qdrant_filter = None

        if source:
            qdrant_filter = models.Filter(
                must=[  # must 所有条件都必须为 True
                    models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))
                ]
            )

        results = self.qdrant_store.similarity_search_with_score(query=query, k=5, filter=qdrant_filter)

        if len(results) == 0:
            print("[rag agent] 未找到相关知识库内容。")
            return "未找到相关知识库内容。"

        for i, (doc, score) in enumerate(results, start=1):
            print(f"检索结果[{i}] score={score:.4f}] doc.metadata={doc.metadata}")
            print(doc.page_content, end="\n\n")

        page_content_list = [doc.page_content for doc, score in results]
        result = json.dumps(page_content_list, ensure_ascii=False)

        print(f'[rag agent] 相似度检索到{len(page_content_list)}个文档', end="\n\n")
        return result

    def metadata_search(self, header_level: str, header_value: str, runtime: ToolRuntime[RagContext]):
        """
        知识库元数据搜索

        Args:
            header_level: 标题层级
            header_value: 标题文本
        """
        # 1. 从自动注入的 config 中安全获取运行时参数,确保只在当前保险条款中检索
        source = runtime.context.source
        print(
            f"[rag agent] 元数据检索 | header_level: {header_level} | header_value: {header_value} | source: {source}")

        # 2. 构建组合过滤条件
        must_conditions = [
            models.FieldCondition(key=f"metadata.{header_level}", match=models.MatchValue(value=header_value)),
            models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))
        ]

        qdrant_filter = models.Filter(must=must_conditions)
        order_by = models.OrderBy(key="metadata.index", direction=models.Direction.ASC)

        # 3. 使用底层 client 的 scroll 方法,直接按 Filter 获取记录(0 向量参与,性能极高)
        records, next_page_offset = self.qdrant_store.client.scroll(
            collection_name=self.collection,
            scroll_filter=qdrant_filter,
            order_by=order_by,
            limit=100,  # 一次性捞取该大章下的前 100 个分片(通常足够容纳一整章)
            with_payload=True,  # 必须为 True,否则拿不到文本
            with_vectors=False  # 不需要返回向量值
        )

        if len(records) == 0:
            print("[rag agent] 未找到相关知识库内容。")
            return f"未找到相关知识库内容"

        # 4. 提取文本(LangChain 存入 Qdrant 时,文本正文默认放在 payload 的 "page_content" 字段中)
        page_content_list = []

        for i, record in enumerate(records, start=1):
            payload = record.payload
            # 打印调试信息,看一眼拿到了哪些片段
            print(f"检索结果[{i}] metadata: {payload.get('metadata')}")
            # print(f"page_content: {payload['page_content']}", end="\n\n")
            page_content_list.append(payload["page_content"])

        # 5. 序列化并返回
        result = json.dumps(page_content_list, ensure_ascii=False)
        print(f'[rag agent] 元数据检索到{len(page_content_list)}个文档', end="\n\n")
        return result

    def get_catalogue(self, runtime: ToolRuntime[RagContext]):
        """获取文档目录结构"""
        print(f"[rag agent] 获取文档目录")
        source = runtime.context.source
        must_conditions = [models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source))]
        qdrant_filter = models.Filter(must=must_conditions)
        order_by = models.OrderBy(key="metadata.index", direction=models.Direction.ASC)

        # 3. 使用底层 client 的 scroll 方法,直接按 Filter 获取记录(0 向量参与,性能极高)
        records, next_page_offset = self.qdrant_store.client.scroll(
            collection_name=self.collection,
            scroll_filter=qdrant_filter,
            order_by=order_by,
            limit=200,
            with_payload=True,  # 必须为 True,否则拿不到文本
            with_vectors=False  # 不需要返回向量值
        )

        print(f'检索到{len(records)}个文档目录')
        metadatas = [record.payload.get("metadata") for record in records]
        for m in metadatas: m.pop("source")  # 删除元数据中的 source 字段，减少数据量

        print(metadatas)
        return json.dumps(metadatas, ensure_ascii=False)


    def run(self, input: str, source: str) -> str:
        print(f'[rag agent] 原始query：{input},source：{source}')

        response = self.agent.invoke(
            input={"messages": [("user", input)]},
            context=RagContext(source=source)
        )

        # 遍历消息历史,拦截所有工具投喂的原始文档片段
        print('-' * 80 + "Rag Agent 消息列表↓↓↓" + '-' * 80)
        for message in response["messages"]:
            print(f'[rag agent]：{type(message)}，{message}', end="\n\n")
        print('-' * 80 + "Rag Agent 消息列表↑↑↑" + '-' * 80, end="\n\n")

        return response["messages"][-1].content


if __name__ == '__main__':
    rag_agent = RagAgent()
    print('RAG Agent 已启动')

    while True:
        query = input('请输入想要检索的内容：')
        if not query: continue

        # 华泰人寿百万康宁终身重大疾病保险 长生爱康保重大疾病保险
        result = rag_agent.run(query, "长生爱康保重大疾病保险")
        print(f"最终回答：{result}")
        print('=' * 100)

    rag_agent.qdrant_store.client.close()