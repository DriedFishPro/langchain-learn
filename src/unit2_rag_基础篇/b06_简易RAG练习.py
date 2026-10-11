from langchain.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_qdrant import QdrantVectorStore

from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit2_rag_基础篇.b02_调用embedding模型 import create_embedding_model
from unit2_rag_基础篇.b03_qdrant基本使用 import init_qdrant_client


# 初始化Qdrant 和 embedding
collection = "rag_b05"
embeddings = create_embedding_model(1024)
qdrant_client = init_qdrant_client(path="./qdrant_db", collection=collection)
qdrant_store = QdrantVectorStore(client=qdrant_client, collection_name=collection, embedding=embeddings)


def similarity_search(query):
    result: list[dict] = []
    docs = qdrant_store.similarity_search_with_score(query=query, k=3)

    for doc, score in docs:
        result.append({
            'page_content': doc.page_content,
            'file': doc.metadata['file'],
            'page': doc.metadata['page']
        })

    # 将检索到的文档内容合并为一个字符串
    return result if result else "未找到相关资料"


def chat(query, docs):
    # 提示词增强
    system_prompt = f"""你是一个知识问答助手，请根据以下资料提供的上下文回答用户问题。
    参考资料：{docs}\n
    请基于以上资料回答用户问题，回答时必须先说明资料出处，如果资料中没有相关信息，请如实告知用户。"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    print(messages, end="\n\n")
    chat_model = create_model()

    for chunk in chat_model.stream(messages):
        print(chunk.text, end='', flush=True)


if __name__ == "__main__":
    query = "动手打人罚款多少？"

    docs = similarity_search(query)
    print(docs)

    chat(query, docs)

    qdrant_store.client.close()