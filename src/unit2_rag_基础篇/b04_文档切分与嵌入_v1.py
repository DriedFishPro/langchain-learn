import os
from uuid import uuid4

from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient

from unit2_rag_基础篇.b02_调用embedding模型 import create_embedding_model
from unit2_rag_基础篇.b03_qdrant基本使用 import init_qdrant_client, test_scroll


# 使用 pypdf 解析 PDF，返回文本列表
def parse_with_pypdf(file_path: str) -> list[dict]:
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    print(f"成功加载 PDF，总页数: {total_pages}")
    result: list[dict] = []

    # 解析每一页的文本内容，并封装为字典
    for i, page in enumerate(reader.pages):
        print(f"正在解析 {os.path.basename(file_path)} 第 {i+1} 页")
        text = page.extract_text()

        if text and text.strip():  # 确保页面有文本，过滤空页
            # 封装时把页码作为 metadata 存入，后续检索时能知道出处
            result.append({
                "page_content": text,
                "metadata": {"file": os.path.basename(file_path), "page": i + 1}
            })

    return result


def split(pages: list[dict]) -> list[dict]:
    print("--- 开始文本切分 ---")
    # 推荐使用递归字符文本切分器，它会尽量保证段落和句子的完整性
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # chunk_size 是每个块的最大字符数
        chunk_overlap=50,  # chunk_overlap 是相邻块的重叠字数（防止语义在切分点断掉）
        separators=["\n", "。", "！", "？", "；", "，", " ", ""],  # 自定义分隔符
    )
    split_result = []

    for i, page in enumerate(pages):
        print(f"正在切分 {i+1}/{len(pages)} {page['metadata']}")
        split_texts = text_splitter.split_text(page["page_content"])

        for j, split_text in enumerate(split_texts):
            split_text = split_text.replace(" ", "")
            split_result.append({"page_content": split_text, "metadata": page["metadata"]})

    return split_result


# 将切分后的文本进行向量化并存入 Qdrant 向量库
def embedding(qdrant_client: QdrantClient, collection: str, embeddings: OpenAIEmbeddings, chunks: list[dict]):
    points: list[PointStruct] = []

    for i, chunk in enumerate(chunks):
        print(f"--正在嵌入 {i + 1}/{len(chunks)} {chunk}", end="\n\n")
        # 调用embedding模型将文本转向量
        vector = embeddings.embed_query(chunk["page_content"])

        point = PointStruct(id=str(uuid4()), vector=vector,  payload=chunk)
        points.append(point)

    operation_info = qdrant_client.upsert(collection_name=collection, points=points, wait=True)
    print(f"数据插入成功！operation_info={operation_info}")


# 进行相似度搜索
def similarity_search(qdrant_client: QdrantClient, query: str, collection: str, embeddings: OpenAIEmbeddings):
    query_vector = embeddings.embed_query(query)
    search_result = qdrant_client.query_points(collection_name=collection, query=query_vector, limit=3)

    print("--- 搜索结果 ---")
    for point in search_result.points:
        print(point, end="\n\n")


if __name__ == '__main__':
    file_path = r".\docs\建筑施工企业安全生产规章制度.pdf"

    # 解析 PDF
    pages = parse_with_pypdf(file_path)
    for page in pages: print(f"len={len(page['page_content'])}, page={page}", end="\n\n")

    # 切分
    chunks = split(pages)
    print(f"切分之后 chunks len={len(chunks)}")
    for chunk in chunks: print(f"len={len(chunk['page_content'])}, chunk={chunk}", end="\n\n")

    # 初始化 Qdrant 和 embedding
    collection = "rag_b04"
    qdrant_client = init_qdrant_client(path='./qdrant_db', collection=collection)
    embedding_model = create_embedding_model(1024)

    # 将文本进行分块、嵌入并写入 Qdrant
    # embedding(qdrant_client, collection, embedding_model, chunks)

    # 进行相似度搜索
    # similarity_search(qdrant_client, "动手打人罚款多少？", collection, embedding_model)

    # test_scroll(qdrant_client, collection)
    # qdrant_client.close()