import operator
import rich

from pathlib import Path
from typing import Annotated, List, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from unit1_大模型基础篇.a03_langchain调用模型 import create_model

# 定义结构化输出
class Section(BaseModel):
    # 章节名称
    name: str = Field(description="章节名称",)
    # 章节概述
    description: str = Field(description="章节概述",)

class Sections(BaseModel):
    # 章节列表
    sections: List[Section] = Field(description="章节列表",)

# Graph state
class State(TypedDict):
    topic: str  # 主题
    sections: list[Section]  # 规划的章节列表
    section_contents: Annotated[list, operator.add]  # 已完成的章节列表


planner_model = create_model(response_format={"type": "json_object"})
structured_planner_model = planner_model.with_structured_output(Sections)
worker_model = create_model()

# 生成报表计划的编排器
def planner(state: State):
    messages = [
        SystemMessage(content="你是一个研究报告规划者，你的任务是根据研究主题，为研究报告生成一个章节列表（数量控制在5个以内），后续将由其它模型负责去为每个章节生成具体的研究报告"),
        HumanMessage(content=f"研究的主题是: {state['topic']}"),
    ]

    report_sections = structured_planner_model.invoke(messages)
    print(f"[planner] 创建计划: {report_sections}", end="\n\n")
    return {"sections": report_sections.sections}

# 每个执行者撰写报告的一部分
def worker(args: dict):
    print(f"[worker] 创建报告片段: {args['section']}")
    messages = [
        SystemMessage(content="你是一个研究报告撰写专家，你的任务是根据当前的研究主题和章节描述，完成本章节的报告内容，字数控制在500以内"),
        HumanMessage(content=f"研究主题: {args['topic']}，章节名称: {args['section'].name}，章节描述: {args['section'].description}"),
    ]

    section = worker_model.invoke(messages)
    print(f"[worker]: {args['section'].name} 已完成", end="\n\n")
    return {"section_contents": [section.content]}

# 将各个片段整合为完整报告
def write_file(state: State):
    # 已完成的片段列表
    section_contents = state["section_contents"]
    # 转为字符串
    completed_report_sections = "\n\n---\n\n".join(section_contents)

    path = Path(__file__).resolve().parent / '研究报告.md'
    path.write_text(completed_report_sections, encoding="utf-8")
    print('[write_file] 报告已生成')

# 动态扇出，利用 Send API 动态派发多个任务并行执行
def fanout(state: State):
    send_list = []

    for section in state["sections"]:
        send = Send("worker", {'topic': state["topic"], 'section': section})
        send_list.append(send)

    print(f'len(send_list)={len(send_list)}')
    return send_list


workflow = StateGraph(State)

workflow.add_node("planner", planner)
workflow.add_node("worker", worker)
workflow.add_node("write_file", write_file)

workflow.add_edge(START, "planner")
workflow.add_conditional_edges("planner", fanout, ["worker"])
workflow.add_edge("worker", "write_file")
workflow.add_edge("write_file", END)

graph = workflow.compile()

result = graph.invoke({"topic": "java和python语言的简单对比"})
rich.print(result)