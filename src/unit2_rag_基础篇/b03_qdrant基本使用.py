from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from unit2_rag_基础篇.b02_调用embedding模型 import create_embedding_model


# 1. 初始化客户端
def init_qdrant_client(path: str, collection: str, size: int = 1024) -> QdrantClient:
    client: QdrantClient = QdrantClient(path=path)
    print("Qdrant客户端已创建")

    # 创建一个名为 "my_collection" 的集合
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=size,  # 向量维度
                distance=Distance.COSINE  # 距离度量
            ),
        )
        print(f"集合创建成功！")
    return client


# 2. 查看集合列表
def show_collections(client: QdrantClient):
    resp = client.get_collections()

    print(f"集合列表：{len(resp.collections)}")
    for c in resp.collections:
        print(c.name)


# 3. 插入数据
def test_upsert(client: QdrantClient, collection: str, embeddings: Embeddings):
    docs = [
        {
            "type": "人工智能",
            "content": "深度学习是机器学习的一个子集，它基于神经网络，尤其是深度神经网络，通过构建多层神经网络模型来模拟人脑神经元的工作方式",
        },
        {
            "type": "宠物",
            "content": "布偶猫是一种体型大、性格温顺、长毛蓝眼的可爱动物",
        },
        {
            "type": "网络流行语",
            "content": "咕咕嘎嘎是网络流行语，原为模拟婴儿声音的拟声词，没有实际含义，后演变为装傻、抽象兴奋的表达",
        },
    ]

    # 准备要插入的数据（Points），包含：唯一的 ID（可以是整数或 UUID）、向量数据（长度必须与创建集合时指定的维度一致）、Payload（元数据）
    points = []

    for index, doc in enumerate(docs):
        vector = embeddings.embed_query(doc["content"])
        point = PointStruct(id=index+1, vector=vector, payload=doc)
        points.append(point)

    operation_info = client.upsert(collection_name=collection, points=points, wait = True)  # wait = True：等待操作在后台完全生效后再返回
    print(f"数据插入成功！operation_info={operation_info}")


# 4. 查询数据
def test_scroll(client: QdrantClient, collection):
    records, next_offset = client.scroll(
        collection,
        with_payload=True, # 返回元数据，默认True
        with_vectors=False # 不返回向量值，默认False
    )

    print(f'len(records)={len(records)}')
    for record in records:
        print(record)


# 5. 根据id查询
def test_retrieve(client: QdrantClient, collection, ids):
    retrieved_points = client.retrieve(collection_name=collection, ids=ids)
    for point in retrieved_points:
        print(f"id查询结果: {point}")


# 6. 向量相似度搜索
def test_query_points(client: QdrantClient, collection, embeddings: Embeddings, query: str):
    # 先把query做向量化
    query_vector = embeddings.embed_query(query)

    # 搜索与 query_vector 最相似的 top 3 数据，如果需要根据元数据过滤，则使用query_filter
    search_result = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=3,  # 返回最相似的前 3 个
        # query_filter=Filter(
        #     must=[
        #         FieldCondition(key="type", match=MatchValue(value="animal"))
        #     ]
        #     # should：多条件“或”关系（OR），must_not：取反/排除条件（NOT）
        # )
    )

    print("--- 向量相似度搜索结果 ---")
    for point in search_result.points:
        print(f"point: {point}")


# 7. 修改元数据
def test_set_payload(client: QdrantClient, collection):
    # set_payload 是局部更新，还有一个 overwrite_payload 方法是全局覆盖
    client.set_payload(
        collection_name=collection,
        payload={"type": "哈基咪"},
        points=[2],  # 指定要修改的 ID 列表
        wait=True
    )

def test_overwrite_payload(client: QdrantClient, collection):
    client.overwrite_payload(
        collection_name=collection,
        points=[2],  # 可以一次传多个 id
        payload={
            "type": "宠物",
            "content": "布偶猫是一种体型大、性格温顺、长毛蓝眼的‌可爱动物"
        },
    )


# 8. 删除数据
def test_delete(client: QdrantClient, collection, ids):
    client.delete(
        collection_name=collection,
        points_selector=ids,  # 删除的 ID 列表
        wait=True
    )


# 9. 删除集合
def test_delete_collection(client: QdrantClient, collection):
    client.delete_collection(collection_name=collection)
    show_collections(client)


if __name__ == '__main__':
    collection = "b_03_cos"
    dimensions = 1024
    # 初始化embedding模型
    embeddings = create_embedding_model(dimensions)

    # 初始化客户端
    client = init_qdrant_client(path="./qdrant_db", collection=collection, size=dimensions)
    # 查看所有集合
    show_collections(client)

    # 测试插入数据
    test_upsert(client, collection, embeddings)
    test_scroll(client, collection)

    # 测试根据id获取数据
    # test_retrieve(client, collection, [1,2])

    # 测试相似度查询
    test_query_points(client, collection, embeddings, "人工智能")

    # 测试修改元数据
    # test_set_payload(client, collection)
    # test_scroll(client, collection)

    # test_overwrite_payload(client, collection)
    # test_scroll(client, collection)

    # 测试删除数据
    # test_delete(client, collection, [3])
    # test_scroll(client, collection)

    # 测试删除集合
    # test_delete_collection(client, collection)

    client.close()
    print("客户端已安全关闭")