from util import env_util
env_util.setup_langsmith()

import json
from langsmith import Client


# 创建数据集
client = Client()
dataset_name = "insurance_dataset"

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="保险业务数据集",
)

# 给数据集添加数据
with open('dataset.json', 'r', encoding='utf-8') as f:
    examples = json.load(f)
    client.create_examples(dataset_id=dataset.id, examples=examples)