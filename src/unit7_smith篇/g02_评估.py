from util import env_util
env_util.setup_langsmith()

import uuid
from pathlib import Path
from pydantic import BaseModel, Field
from langsmith import Client

from unit4_智能保险客服助手_v1.agent.master_agent import MasterAgent
from unit6_智能保险客服助手_v2.graph.master_graph import MasterGraph
from unit1_大模型基础篇.a03_langchain调用模型 import create_model


class EvaluationResult(BaseModel):
    score: int = Field(description="针对回答的准确性打分，范围 0 到 100")
    reason: str = Field(description="给出打分的判定依据，说明哪些要点命中、哪些缺失或冲突")


# 创建评估模型
eva_model = create_model(model="qwen3.8-max", temperature=0, response_format={"type": "json_object"})
structured_eva_model = eva_model.with_structured_output(EvaluationResult)

# 创建MasterAgent
agent_builder = MasterGraph()
agent = agent_builder.build_graph()

metadata = {
    "master": {
        "model_name": agent_builder.generate_model.model_name,
        "temperature": agent_builder.generate_model.temperature,
        # "thinking": agent_builder.model.extra_body['enable_thinking']
    },
    "eva": {
        "model_name": eva_model.model_name,
        "temperature": eva_model.temperature,
        # "thinking": eva_model.extra_body['enable_thinking']
    },
}


# 定义目标函数
def target(inputs: dict) -> dict:
    thread_id = str(uuid.uuid4())
    print(f'target [{thread_id}] inputs={inputs}')

    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke(input={"messages": [("user", inputs["question"])]}, config=config)
    result = response["messages"][-1].content

    print(f'target [{thread_id}] result={result}')
    return {"answer": result}

# 定义评估器（准确性评估）
def evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    input：来自于数据集中的inputs
    outputs: 目标函数运行返回的字典
    reference_outputs: 数据集中的outputs
    """
    print(f"\n{'='*50} [evaluator] 开始↓↓↓ {'='*50} ")
    print("[evaluator] 问题:", inputs, end='\n\n')
    print("[evaluator] 参考答案:", reference_outputs, end='\n\n')
    print("[evaluator] agent实际回答:", outputs, end='\n\n')

    prompt_template = (Path(__file__).resolve().parent / "eva_prompt.md").read_text(encoding="utf-8")
    prompt = (prompt_template.replace('{question}', inputs["question"])
                .replace('{reference_answer}', reference_outputs["answer"])
                .replace('{key_point}', reference_outputs["key_point"])
                .replace('{llm_answer}', outputs['answer']))

    response = structured_eva_model.invoke(prompt)
    print(f'evaluator 评估结果：{response}')     # response 直接就是一个 EvaluationResult 对象
    print(f"{'=' * 50} evaluator 结束↑↑↑ {'=' * 50} ", end='\n\n')

    # 我们不仅把 score 传给 LangSmith，还可以把理由塞进 comment 里，方便在 UI 中查看
    return {"key": "accuracy", "score": response.score / 100.0, "comment": response.reason}

if __name__ == "__main__":
    client = Client()
    experiment_results = client.evaluate(
        target,  # 被评估的目标函数
        data='insurance_dataset',  # 数据集名称
        evaluators=[evaluator],  # 评估器列表
        experiment_prefix="graph_experiment",  # 评估后生成的实验报告的前缀
        description="graph版第1次评估",  # 实验描述，用来记录这次测试的背景
        max_concurrency=1,  # 最大并发数，可选参数。如果你的数据集很大（比如 1000 条数据），一条一条去请求 LLM 会非常慢
        metadata=metadata,  # 可选参数，实验元数据
    )
    print(experiment_results)