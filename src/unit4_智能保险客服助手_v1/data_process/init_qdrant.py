from pathlib import Path

import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from unit2_rag_基础篇.b02_调用embedding模型 import create_embedding_model


# 加载并切分文件
def load_split_mdfile(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

        # 根据哪些标题切分，切分后，对应层级的标题会自动挪到 Document 的 metadata 中
        headers_to_split_on = [
            ("#", "header1"),  # 一级标题 → 大章节
            ("##", "header2"),  # 二级标题 → 小节
            ("###", "header3"),  # 三级标题 → 细分点
        ]

        # 初始化轻量化的 Markdown 头部切分器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            # strip_headers=False  # 是否去除正文的标题，False：不去除，默认True，去除
        )

        # split_text 方法会直接返回一个 list[Document]
        docs: list[Document] = markdown_splitter.split_text(markdown_content)

        # 顺手把文件路径补进元数据（Metadata）里，方便后续做 RAG 溯源
        path = Path(file_path)

        for index, doc in enumerate(docs):
            doc.metadata["source"] = path.stem  # 文件名（产品名），用于数据隔离
            doc.metadata['index'] = index  # 用于检索排序，以一个有序的列表发给大模型，更容易理解

            # 从doc的元数据中提取出三级标题作为面包屑导航
            breadcrumbs = []  # ['第七章 附表', '附表一：重大疾病种类表', '脑中风后遗症']

            for level in ["header1", "header2", "header3"]:
                if level in doc.metadata: breadcrumbs.append(doc.metadata[level])

            if breadcrumbs:
                breadcrumb_str = " / ".join(breadcrumbs)  # 拼接后：第七章 附表 / 附表一：重大疾病种类表 / 脑中风后遗症
                doc.page_content = f"【{breadcrumb_str}】\n{doc.page_content}"  # 将面包屑导航加到正文前面

        print(f"成功加载并切分出 {len(docs)} 个文档块")
        return docs

def init_qdrant_store(collection: str, path: str) -> QdrantVectorStore:
    print(f"初始化 Qdrant path：{path}")
    qdrant_client = QdrantClient(path=path)
    # 实例化 embedding 模型，用于语义检索
    embeddings = create_embedding_model()
    # 实例化本地 BM25 稀疏向量模型，用于关键词检索
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # 创建集合
    if not qdrant_client.collection_exists(collection):
        qdrant_client.create_collection(
            collection_name=collection,
            # 稠密向量配置（保持原有的 size 和 distance）
            vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
            # 必须新增稀疏向量配置，否则无法存储 BM25 矩阵
            sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
        )
        print(f"已创建 {collection} 集合")

    qdrant_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,     # 绑定稀疏向量模型
        retrieval_mode=RetrievalMode.HYBRID,    # 显式声明为混合检索
        vector_name="dense",                    # 稠密向量空间的名称
        sparse_vector_name="sparse"             # 稀疏向量空间的名称
    )
    return qdrant_store


if __name__ == '__main__':
    name_list = ["长生爱康保重大疾病保险", "华泰人寿百万康宁终身重大疾病保险"]
    file_list = [r"../doc/长生爱康保重大疾病保险.md", r"../doc/华泰人寿百万康宁终身重大疾病保险.md"]

    collection = "unit4_rag"
    qdrant_store = init_qdrant_store(collection, "../db/qdrant")
    option = input("请选择操作：1.灌库；2.猫一眼；3.检索测试\n")

    if option == "1":   # 开始灌库
        for file_path in file_list:
            docs = load_split_mdfile(file_path)

            for i, doc in enumerate(docs):
                print(f'{i} metadata：{doc.metadata}')
                print(doc.page_content, end='\n\n')

            # 将文档块进行向量化并写入 Qdrant，忽略第一个文档块（产品名称）
            qdrant_store.add_documents(docs[1:])

    elif option == "2":     # 写入后检查一下qdrant中的数据
        enc = tiktoken.encoding_for_model("gpt-4o")
        index = input(f"0-[{name_list[0]}]，1-[{name_list[1]}]，要看哪个文档？\n")

        filter = Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=name_list[int(index)]))])
        order_by = models.OrderBy(key="metadata.index", direction=models.Direction.ASC)
        records, next_offset = qdrant_store.client.scroll(collection, limit=1000, scroll_filter=filter, order_by=order_by)

        for i, record in enumerate(records, 1):
            tokens = enc.encode(record.payload['page_content'])
            text_len = len(record.payload['page_content'])
            print(f'{i} 文本长度：{text_len}，token长度：{len(tokens)}，比例：{(len(tokens) / text_len):.2f}')
            print(record, end="\n\n")

    elif option == "3":
        index = input(f"0-[{name_list[0]}]，1-[{name_list[1]}]，要检索哪个文档？\n")
        query = input(f"输入检索关键词：\n")

        filter = Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=name_list[int(index)]))])
        results = qdrant_store.similarity_search_with_score(query=query, filter=filter, k=5)

        for doc, score in results:
            print(f"[得分: {score:.4f}] 元数据: {doc.metadata}")
            print(doc.page_content, end='\n\n')

    qdrant_store.client.close()