import os
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from unit2_rag_基础篇.b02_调用embedding模型 import create_embedding_model
from unit2_rag_基础篇.b03_qdrant基本使用 import init_qdrant_client


# 使用 pypdf 解析 PDF，并转化为 LangChain 的 Document 对象列表
def parse_with_pypdf(file_path: str) -> list[Document]:
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    print(f"成功加载 PDF，总页数: {total_pages}")
    docs: list[Document] = []

    # 解析每一页的文本内容，并封装为 LangChain 的 Document
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():  # 确保页面有文本，过滤空页
            # 封装时把页码作为 metadata 存入，后续检索时能知道出处
            doc = Document(page_content=text, metadata={"file": os.path.basename(file_path), "page": i + 1})
            docs.append(doc)

    return docs


# 将解析后的文本进行分块、嵌入并存入 Qdrant 向量库
def split_and_embedding(qdrant_store: QdrantVectorStore, docs: list[Document]):
    print("--- 开始文本切分 ---")
    # 推荐使用递归字符文本切分器，它会尽量保证段落和句子的完整性
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # chunk_size 是每个块的最大字符数
        chunk_overlap=50,    # chunk_overlap 是相邻块的重叠字数（防止语义在切分点断掉）
        separators = ["\n", "。", "！", "？", "；", "，", " ", ""]
    )

    # 执行切分
    split_docs = text_splitter.split_documents(docs)
    print(f"切分完成，共生成 {len(split_docs)} 个文本块。")

    # 将切分后的文档自动计算向量并写入库中
    qdrant_store.add_documents(split_docs)
    print(f"所有文本块已成功嵌入并存入 Qdrant 向量库")


# 进行相似度搜索
def similarity_search(qdrant_store: QdrantVectorStore, query: str) -> list[str]:
    results = qdrant_store.similarity_search_with_score(query=query, k=5)

    for doc, score in results:
        print(f"[score={score:.4f}] {doc}", end="\n\n")


if __name__ == '__main__':
    file_path = r".\docs\建筑施工企业安全生产规章制度.pdf"

    # 解析 PDF
    docs = parse_with_pypdf(file_path)

    # 初始化 Qdrant 和 embedding
    collection = "rag_b05"
    qdrant_client = init_qdrant_client(path='./qdrant_db', collection=collection)
    embedding_model = create_embedding_model(1024)
    qdrant_store = QdrantVectorStore(client=qdrant_client, collection_name=collection, embedding=embedding_model)

    # 将文本进行分块、嵌入并写入 Qdrant
    split_and_embedding(qdrant_store, docs)

    # 进行相似度搜索
    similarity_search(qdrant_store, "动手打人罚款多少？")

    qdrant_store.client.close()