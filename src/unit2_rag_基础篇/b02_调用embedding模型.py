import os

from langchain.embeddings import init_embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from unit2_rag_基础篇.b01_向量相似度 import cosine_similarity

from util import env_util
env_util.setup_apikey()


def create_embedding_model(dimensions: int = 1024) -> Embeddings:
    """
    避坑点：使用 langchain_openai 库时，LangChain 底层会把你的文本切分成 batch_tokens（一串数字 Token ID），然后发送给 API。
    但是，阿里云百炼的 compatible-mode（兼容模式）在处理 Embedding 时，不接受 Token ID 数组作为输入，它只接受纯文本字符串（str）或纯文本列表（list of str）
    要将check_embedding_ctx_length设为false，关闭本地的 token 长度检查和切片，直接传纯文本
    """

    embeddings = init_embeddings(
        model="text-embedding-v4",  # 你的模型名称
        provider="openai",  # 因为兼容 OpenAI 接口，所以 provider 指定为 openai
        base_url=f"https://{os.getenv("DASHSCOPE_WORKSPACE_ID")}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        dimensions=dimensions,    # 向量维度
        chunk_size=10,  # 避坑点：限制每次提交给 Embedding 接口的文本数量不超过 10
        check_embedding_ctx_length=False  # 关闭本地的 token 长度检查和切片
    )
    return embeddings


def create_openai_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # dimensions=1024,    # 向量维度
        chunk_size=10,  # 避坑点：限制每次提交给 Embedding 接口的文本数量不超过 10
        check_embedding_ctx_length=False
    )
    return embeddings


# 2. 调用 embed_query 处理单个查询
def test_embed_query(embeddings: OpenAIEmbeddings, query: str):
    query_result: list[float] = embeddings.embed_query(query)
    print(f"type={type(query_result)}，len={len(query_result)}")
    print(f"前5个值: {query_result[:5]}")
    return query_result


# 3. 调用 embed_documents 批量处理文档列表
def test_embed_documents(embeddings: OpenAIEmbeddings, doc_list: list[str]):
    doc_results: list[list[float]] = embeddings.embed_documents(doc_list)

    for i, doc_result in enumerate(doc_results):
        print(f"文档 {i+1}，向量维度: {len(doc_result)}")
        print(f"前5个值: {doc_result[:5]}")

    return doc_results


if __name__ == '__main__':
    embeddings = create_embedding_model(1024)

    # 使用embedding模型来将猫猫狗狗转化为真正的向量，看看是否符合我们的预期
    vec_cat = test_embed_query(embeddings, "小猫")
    vec_dog = test_embed_query(embeddings, "小狗")
    vec_apple = test_embed_query(embeddings, "苹果")
    vec_banana = test_embed_query(embeddings, "香蕉")

    # 计算余弦相似度
    sim1 = cosine_similarity(vec_cat, vec_dog)
    print(f"猫-狗 相似度: {sim1:.4f}")

    sim2 = cosine_similarity(vec_cat, vec_apple)
    print(f"猫-苹果 相似度: {sim2:.4f}")

    sim3 = cosine_similarity(vec_apple, vec_banana)
    print(f"苹果-香蕉 相似度: {sim3:.4f}")


    # 长句子测试
    query = test_embed_query(embeddings, "人工智能")

    docs = [
        "深度学习是机器学习的一个子集，它基于神经网络，尤其是深度神经网络，通过构建多层神经网络模型来模拟人脑神经元的工作方式",
        "布偶猫是一种体型大、性格温顺、长毛蓝眼的‌可爱动物",
        "咕咕嘎嘎是网络流行语，原为模拟婴儿声音的拟声词，没有实际含义，后演变为装傻、抽象兴奋的表达"
    ]

    doc_results = test_embed_documents(embeddings, docs)

    # 相似度计算
    for i, doc in enumerate(doc_results):
        similarity = cosine_similarity(query, doc)
        print(f"文档 {i + 1} 的相似度: {similarity}")