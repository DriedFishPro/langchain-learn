import numpy as np

def cosine_similarity(v1, v2):
    """
    计算两个向量的余弦相似度
    公式： (A·B) / (||A|| * ||B||)
    """
    # 点乘：对应位置相乘后相加
    dot_product = np.dot(v1, v2)
    # 求向量的模长（长度）
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    return dot_product / (norm_v1 * norm_v2)


if __name__ == '__main__':
    # 假设这是经过某种魔法转换后的2维向量，维度假设为：[动物特征（会跑）, 食物特征（能吃）]
    vec_dog = np.array([0.9, 0.3])  # 狗：擅长奔跑，少部分地区会吃狗肉
    vec_cat = np.array([0.8, 0.1])  # 猫：擅长跑酷，基本不作为食物
    vec_apple = np.array([0.1, 0.9])  # 苹果：几乎不会动，是典型食物
    vec_banana = np.array([0.05, 0.85])  # 香蕉：几乎不会动，是典型食物

    print(f"猫-狗 点积：{np.dot(vec_cat, vec_dog)}")
    print(f"猫-苹果 点积：{np.dot(vec_cat, vec_apple)}")

    # 计算相似度
    sim1 = cosine_similarity(vec_dog, vec_cat)
    print(f"猫-狗 相似度: {sim1:.4f}")  # 较高，符合预期，都是动物

    sim2 = cosine_similarity(vec_apple, vec_banana)
    print(f"苹果-香蕉 相似度: {sim2:.4f}")  # 较高，符合预期，都是食物

    sim3 = cosine_similarity(vec_cat, vec_apple)
    print(f"猫-苹果 相似度: {sim3:.4f}") # 较低，符合预期，一个是动物，一个是食物
