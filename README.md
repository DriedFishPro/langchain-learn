# LangChain 构建 AI Agent 光速入门教程

## 前言
### 为什么创建这套教程？
我本是一个java开发者，从去年开始接触Agent开发，从我自己的学习经历讲，我发现现有的学习资源存在以下问题：

1. **教程太长**：动辄十几二十小时（还只是LangChain或者LangGraph其中一个专题），节奏较慢，一个环境搭建就得讲上个把小时
2. **内容不全面**：大量教程要么只讲LangChain的，要么只讲LangGraph的，很少有讲LangSmith的，往往要找好几个教程才能凑齐完整内容
3. **标题党太多**：打着"绝对是B站No.1"的旗号，点进去发现，内容是不知道从哪搬运来东拼西凑的
4. **官方文档学习门槛较高**：生态庞大，官方文档有 DeepAgent、LangChain、LangGraph、LangSmith等众多模块，初学者很容易知识过载，迷失方向

基于上述痛点，我决定把自己的学习经过制作为教程，希望能帮助到更多想学习Agent开发的小伙伴们。

--- 

## 教程定位
本课程旨在打造一套**精炼、深入、实战导向**的 LangChain 快速入门教程

- ✅ **最新版本**：基于 LangChain v1.3+ 讲解
- ✅ **精炼高效**：聚焦核心功能，避免信息过载，快速上手，尽量以最短的时间帮助大家建立相对完整的知识体系
- ✅ **深度揭秘**：通过抓包分析，揭示 LLM 工具绑定、结构化输出等底层的数据交互机制
- ✅ **原理剖析**：通过"手搓"的方式，不依赖框架，自己实现工具绑定、ReAct循环、HITL等核心机制
- ✅ **项目练习**：设计一个贯穿整个课程的练习项目，每章知识点讲完后就实现一部分功能，循序渐进构建完整应用
- ✅ **内容丰富**：涵盖了从基础到高级知识点讲解，配套的项目练习，以及观测、评估、部署等一套完整的agent开发体系内容

如果你希望快速掌握用 LangChain 来构建 Agent，那么本教程将是你的理想选择。

--- 

## 前置要求
### python基础
本课程用的是python语言，所以最好得有python基础（或者其它编程语言基础），
但并非只有python才能做agent应用，实际上 LangChain 也有java版、js版

### 大语言模型基础
1. 理解 LLM、Token、Prompt、temperature 等基本概念
2. 熟悉主流模型提供商（OpenAI、Anthropic、阿里云百炼、硅基流动、DeepSeek 等）

---

## 环境准备
- python 3.12 + uv
- langchain1.3+
- 阿里云百炼平台api key
- ollama（可选）

### 安装依赖
1. 安装好python环境后，在本地安装uv包管理器：`pip install uv`
2. 把代码克隆到本地后，在pycharm中（或其它ide）打开
3. 在项目根目录执行：`uv sync` 安装依赖


### api_key配置
本工程中为了省事，没有用dotenv，而是直接通过代码的方式去设置环境变量的，所有的环境变量都在[env_util.py](./util/env_util.py)中配置

---

## LangChain 是什么？
从广义上讲，LangChain 不仅仅是一个框架，而是一个完整的 Agent 应用开发生态体系，由四大核心组件构成：

### 🔄 LangGraph（Runtime）
LangGraph是agent的底层运行时，它提供了agent运行时所需要的基础设施：状态管理、数据持久化、人机交互（HITL）等，
同时它也是一套基于图结构的流程编排框架，你可以用它来以工作流的形式构建agent。

### 📦 LangChain（Framework）
LangChain 是在 graph 基础之上构建的一个高度可配置的 Agent 开发框架，它的核心功能是提供一个简洁的 create_agent api，
将大模型（LLM）与提示词、工具以及各类中间件组合在一起，形成一个完整的agent执行循环，你可以用它来快速构建出具备工具调用能力的标准 Agent。

### 🤖 DeepAgent（Harness）
DeepAgent 是在 LangChain 基础上进一步封装的 Agent 开发套件，它内置了任务规划、上下文管理、子智能体、文件操作等常用功能，
你无需自己去配置，就能快速构建出一个具备复杂任务处理能力的Agent。

### 🔍 LangSmith（DevOps）
Agent应用的DevOps平台，它帮助开发者追踪应用执行过程、分析性能瓶颈、建立自动化评估体系，确保 Agent 应用的稳定性和可靠性。

> 注：本教程暂不包含DeepAgent的相关内容，我还没去仔细研究，只是大概看了一下，后面也许会再出一个DeepAgent、MCP、Skill这些内容的教程

---

## 课程大纲
### [第一章：大模型基础篇](./src/unit1_大模型基础篇/unit1_readme.md)
#### 1.1 如何调用大模型
#### 1.2 大模型的消息类型及格式
#### 1.3 大模型的结构化输出
#### 1.4 大模型调用工具原理
#### 1.5 天气查询助手练习

### [第二章：rag基础篇](./src/unit2_rag_基础篇/unit2_readme.md)
#### 2.1 RAG的概念
#### 2.2 向量的概念
#### 2.3 embedding模型
#### 2.4 向量库的基本使用（qdrant）
#### 2.5 文档切分与嵌入
#### 2.6 简易rag项目练习

### [第三章：agent基础篇](./src/unit3_agent_基础篇/unit3_readme.md)
#### 3.1 Agent的概念
#### 3.2 Agent的工作模式
#### 3.3 create_agent快速上手
#### 3.4 @tool装饰器
#### 3.5 streaming（流式输出）
#### 3.6 event streaming（事件流）
#### 3.7 结构化输出
#### 3.8 text2sql项目实战

### [第四章：agent高级篇](./src/unit4_agent_高级篇/unit4_readme.md)
#### 4.1 中间件（装饰器版）
#### 4.2 中间件（class版）
#### 4.3 短期记忆
#### 4.4 长期记忆
#### 4.5 人机交互（HITL）
#### [4.5 智能保险客服助手_v1](./src/unit4_智能保险客服助手_v1/unit4_智能保险客服助手_v1)

### [第五章：graph基础篇](./src/unit5_graph_基础篇/unit5_readme.md)
#### 5.1 Graph快速上手
#### 5.2 Graph运行模式
#### 5.3 checkpoint
#### 5.4 store

### [第六章：graph高级篇](./src/unit6_graph_高级篇/unit6_readme.md)
#### 6.1 重试机制
#### 6.2 超时机制
#### 6.3 异常处理
#### 6.4 中断机制（HITL）
#### [6.5 智能保险客服助手_v2](./src/unit6_智能保险客服助手_v2/unit6_智能保险客服助手_v2.md)

### [第七章：smith篇](./src/unit7_smith篇/unit7_readme.md)
#### 7.1 链路追踪
#### 7.2 创建数据集
#### 7.3 使用数据集评估

### [第八章：Web UI与部署篇](./src/unit8_webui与部署篇/unit8_readme.md)
#### 8.1 服务化运行
#### 8.2 构建agent-ui
#### 8.3 容器化部署