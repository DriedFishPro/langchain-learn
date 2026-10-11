# 第八章：Web UI与部署篇
本章将介绍如何将我们前面开发的**智能保险客服助手**Agent部署到服务端，并使用 Vue.js 创建一个前端工程来连接这个服务进行对话，最终效果如下：
![webui](./image/webui.png)


## 8.1 服务化运行
在前面的学习中，我们都是通过main方法中调用invoke来运行agent程序的，这种方式属于本地执行，它没有对外提供服务的能力，为了让前端工程可以连接agent，我们需要将agent应用部署为服务。

LangSmith不仅提供了观测和评估，还提供了服务构建和部署的能力，通过langgraph-cli可以实现快速构建和部署agent服务。

### 8.1.1 构建langgraph.json配置文件
`langgraph.json` 是 LangGraph 项目的核心配置文件，它在概念上与前端工程中的 `package.json` 类似。

```json
{
  "dependencies": ["../.."],
  "env": ".env",
  "graphs": {
    "agent": "./agent_server.py:build_agent",
    "graph": "./agent_server.py:build_graph"
  }
}
```

- dependencies：指定agent程序需要加载的本地模块或依赖包，通常填写`requirements.txt`或`pyproject.toml`文件所在目录即可
  - ".."表示 langgraph.json 配置文件所在目录的上一级父目录
- graphs：agent程序的入口，指向具体的代码文件及编译好的graph实例，可配置多个。
- env：指定本地开发时加载的环境变量文件路径。
- checkpointer（可选）：配置短期记忆。
- store（可选）：配置长期记忆。

**env：**
```text
DASHSCOPE_API_KEY=sk-3384dc71ec924ef1b50da80677ee10da
OPENAI_API_KEY=sk-tVgfqJD1eI0lUL9mAa0RsCjZgjdXX9f3FhT1uA8IrDQ4A2Uw

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=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
LANGSMITH_PROJECT=langchain-learn
```

**agent_server：**
```python
from functools import lru_cache
from unit4_智能保险客服助手_v1.agent.master_agent import MasterAgent
from unit6_智能保险客服助手_v2.graph.master_graph import MasterGraph


# "agent": "./agent_server.py:build_agent",
@lru_cache(maxsize=1)
def build_agent():
    agent_builder = MasterAgent()
    return agent_builder.build_agent()

# "graph": "./agent_server.py:build_graph"
@lru_cache(maxsize=1)
def build_graph():
    graph_builder = MasterGraph()
    return graph_builder.build_graph()
```

**注意事项：**
> 使用 langgraph dev 启动服务时，build_agent 方法会被调用多次，
> 这是因为 LangGraph CLI / 运行时会在不同的生命周期阶段出于不同目的对其进行多次加载和探测
> 
> 而本地模式的 Qdrant 不支持多个 Client 实例同时读写（即：同一个磁盘目录在同一时间内只允许一个 Client 实例），
> 为了防止 Qdrant 报错，所以这里用了 lru_cache 来做缓存，避免出现多个 Client 实例

更多配置项请参考官方文档：https://docs.langchain.com/langsmith/cli#configuration-file


### 8.1.2 启动本地服务
使用 `langgraph dev` 命令来启动本地agent服务，无需安装 Docker，适合开发和测试，注意：必须要先进入到 `langgraph.json` 文件所在目录下运行该命令  

```shell
langgraph dev --port 2024 --allow-blocking
```
> allow-blocking：允许在代码中执行同步的 I/O 阻塞操作，否则本地的sqlite、qdrant都无法操作了  
> 更多参数请参考官方文档：https://docs.langchain.com/langsmith/cli#dev

![命令行](./image/terminal.png)

启动成功后会自动打开langsmith的studio页面，你可以在这里去测试agent服务

![调试](./image/smith_studio.png)

---

## 8.2 构建agent-ui
agent应用以服务的方式运行之后，就可以通过 api 来访问了，本节的两个核心知识点是：
- Agent Server API
- 前端 SDK（Vue / React）

### 8.2.1 初始化工程
使用 Vite 创建 Vue3 工程，并安装核心依赖：

```bash
npm create vue@latest agent-ui
cd agent-ui
npm install @langchain/vue pinia marked dompurify
```

核心依赖说明：
- `@langchain/vue`：LangChain 官方的 Vue SDK
- `pinia`：Vue 状态管理，用于在组件间共享当前会话 threadId
- `marked` + `dompurify`：将 AI 回复的 Markdown 文本渲染为 HTML 并做 XSS 净化

工程结构（仅列出核心文件）：
```text
src/
├── views/
│   ├── chat_layout.vue      # 整体布局（左侧边栏 + 右聊天区）
│   ├── ChatSidebar.vue      # 会话列表管理
│   └── ChatMain.vue         # 聊天主界面（核心）
├── stores/store.js          # Pinia 全局状态
├── utils/request.js         # HTTP 请求封装
└── components/
    └── MarkdownRenderer.vue # Markdown 渲染组件
```

### 8.2.2 会话管理（ChatSidebar.vue）
Agent Server 内置了 Thread（会话）管理的 REST API，前端通过 HTTP 请求直接调用：

| 功能     | 接口                    | 方法     |
|--------|-----------------------|--------|
| 查询会话列表 | `/threads/search`     | POST   |
| 删除会话   | `/threads/{threadId}` | DELETE |

> 更多 server api 参考官方文档：https://docs.langchain.com/langsmith/server-api-ref

**加载会话列表：**  
```js
async getSessionList(){
  this.sessionList = await post('/threads/search', {})
},
```

**新建会话：**  

使用 `crypto.randomUUID()` 生成一个 UUID 作为新的 threadId，设置到全局状态中

```js
handleNewChat() {
  const uuid = crypto.randomUUID()
  console.log('handleNewChat', uuid)
  this.chatStore.setThreadId(uuid)
},
```

**切换会话：**  

点击历史会话时，将对应的 threadId 设置到全局状态中

```js
handleSelectThread(threadId) {
  this.chatStore.setThreadId(threadId)
},
```

**删除会话：**
```js
async handleDeleteThread(threadId) {
  await del(`/threads/${threadId}`)
  await this.getSessionList()
},
```

**组件刷新机制：**  
> 在 `chat_layout.vue` 中通过 `:key="chatStore.currentThreadId"` 绑定 key，当 threadId 变化时 Vue 会销毁并重建 ChatMain 组件，从而触发 useStream 重新初始化连接。


### 8.2.3 对话主界面（ChatMain.vue）
`useStream` 是 `@langchain/vue` 提供的核心组合式函数，它封装了与 LangGraph Server 的流式通信协议，自动处理 SSE 流式响应的解析和消息状态的维护。

**基本用法：**
```js
import { useStream } from '@langchain/vue'

const stream = useStream({
  apiUrl: 'http://localhost:2024',   // Agent 服务地址
  assistantId: 'agent',              // 对应 langgraph.json 中 graphs 的 key
  threadId: 'xxx-uuid-xxx',          // 会话ID，用于多轮对话记忆
})
```

参数说明：
- `apiUrl`：LangGraph Server 的地址，本地开发时为 `langgraph dev` 启动的端口
- `assistantId`：要调用的 graph 名称，对应 `langgraph.json` 中 `graphs` 字段的 key
- `threadId`：会话标识，如果传入新的 threadId，则会自动开启一个新会话，如果传入旧的 threadId，则会自动加载该会话的历史消息

**useStream 返回的核心属性和方法：**

| 属性/方法           | 类型     | 说明                  |
|-----------------|--------|---------------------|
| `messages`      | Ref    | 当前会话的消息列表，自动响应式更新   |
| `isLoading`     | Ref    | Agent 是否正在处理/输出中    |
| `submit(input)` | Method | 向 Agent 发送消息并触发流式响应 |
| `error`         | Ref    | 错误信息                |

> 更多用法请参考官方文档：https://reference.langchain.com/javascript/langchain-vue/use-stream


**与服务端建立连接**

在 Vue 组件的 `setup()` 中初始化 useStream，然后通过 computed 暴露响应式状态：

```js
export default {
  setup() {
    const chatStore = useChatStore()
    const stream = useStream({
      apiUrl: 'http://localhost:2024',
      assistantId: 'agent',
      threadId: chatStore.currentThreadId,
    })
    return { stream }
  },

  computed: {
    messages() { return this.stream.messages.value },
    isLoading() { return this.stream.isLoading.value },
    error() { return this.stream.error.value },
  },
}
```

**渲染消息列表**

模板中遍历 `messages` 渲染对话气泡，根据 `msg.type` 区分消息角色：
- `human`：用户消息
- `ai`：AI 回复
- `tool`：工具调用结果

```html
<template v-for="msg in messages" :key="msg.id || msg.type + Math.random()">
  <div v-if="getMessageText(msg.content).trim()" class="msg-item" :class="msg.type">
    <div class="msg-avatar">
      {{ isHuman(msg) ? '🧑' : (isAI(msg) ? '🤖' : '🔧') }}
    </div>
    <div class="msg-content">
      <MarkdownRenderer :content="getMessageText(msg.content)" />
    </div>
  </div>
</template>
```


**发送消息：**
```js
async sendMessage() {
  const text = this.userInput.trim()
  if (!text || this.isLoading) return

  await this.stream.submit({
    messages: [{ type: 'human', content: text }]
  })

  this.userInput = ''
  this.resetTextareaHeight()

  const chatStore = useChatStore()
  chatStore.triggerRefresh()
},
```
调用 `submit` 后，useStream 内部会自动：
1. 将用户消息追加到 `messages` 列表
2. 通过 SSE 流式请求 Agent 服务
3. 实时将 AI 的响应增量更新到 `messages` 中
4. 请求结束后自动将 `isLoading` 置为 false

### 8.2.4 全局状态（Pinia Store）
使用 Pinia 在侧边栏和聊天区之间共享会话状态：

```js
export const useChatStore = defineStore('chat', () => {
  const currentThreadId = ref(null)     // 当前会话ID
  const refreshFlag = ref(0)            // 刷新信号

  function setThreadId(id) { currentThreadId.value = id }
  function triggerRefresh() { refreshFlag.value++ }

  return { currentThreadId, setThreadId, refreshFlag, triggerRefresh }
})
```

- `currentThreadId`：当前选中的会话ID，ChatMain 和 ChatSidebar 共同依赖
- `refreshFlag`：每次发送消息后递增，通知侧边栏刷新会话列表（因为新会话首次发消息后才会出现在列表中）


### 8.2.5 启动前端工程
```bash
# 先安装依赖
npm install

# 启动前端工程
npm run dev
```
启动后访问 `http://localhost:5173`，确保 Agent 服务（`langgraph dev --port 2024`）已先行启动，即可开始对话。

**扩展（UI 组件库）：**
> 我们的ChatMain.vue中，对话框是自己实现的，LangChain也提供了几种不同风格的组件库  
> 组件库参考：https://docs.langchain.com/oss/javascript/langchain/frontend/integrations/overview

---

## 8.3 容器化部署

### 8.3.1 前置准备
1. 在langgraph.json中添加python版本号
    ```json
    "python_version": "3.12",
    ```
2. 修改pyproject.toml
    ```toml
    [build-system]
    requires = ["setuptools>=61.0.0", "wheel"]
    build-backend = "setuptools.build_meta"
    
    [tool.setuptools.packages.find]
    where = ["src"]
    namespaces = true
    include = [
        "unit1_大模型基础篇*",
        "unit2_rag_基础篇*",
        "unit3_agent_基础篇*",
        "unit4_智能保险客服助手_v1*",
        "unit6_智能保险客服助手_v2*",
        "unit8_webui与部署篇*",
        "util*"
    ]
    ``` 
    > 注意：authors 中的 email 不能为空，否则打包或报错

### 8.3.2 构建agent镜像
容器中的 agent 服务需要依赖`postgres`和`redis`来做数据持久化，如果你的docker中已有`postgres`和`redis`，那么只需要构建agent镜像即可

- 构建命令
  ```shell
  langgraph build --tag my-agent
  ```

- 运行命令
  ```text
  docker run -d --name my-agent 
  --env-file .env 
  -e REDIS_URI=redis://host:port 
  -e DATABASE_URI=postgres://postgres:password@host:port/name
  -p 2024:8000 
  my-agent:latest
  ```
  
**Docker Engine 镜像源**
```json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "dns": [
    "223.5.5.5",
    "119.29.29.29"
  ],
  "experimental": false,
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ]
}
```

### 8.3.3 启动服务集群
如果你的docker中没有`postgres`和`redis`，那就用这个 docker-compose 来启动整个服务集群

```yaml
name: my-agent-stack

services:
  # 1. PostgreSQL 数据库服务
  postgres:
    image: postgres:18
    container_name: postgres
    restart: always
    environment:
      POSTGRES_PASSWORD: "123456"  # 请替换为你的数据库密码
      POSTGRES_DB: "langgraph"                     # 默认数据库名
    ports:
      - "5432:5432"
    volumes:
      - ./data/postgres:/var/lib/postgresql       # 使用相对路径挂载数据持久化

  # 2. Redis 缓存服务
  redis:
    image: redis:8.8
    container_name: redis
    restart: always
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data                        # 使用相对路径挂载数据持久化

  # 3. Agent 服务容器
  agent:
    image: my-agent:latest                       # 对应 langgraph build --tag my-agent 构建的镜像
    container_name: my-agent
    restart: always
    ports:
      - "2024:8000"                               # 将宿主机的 2024 映射到容器内 LangGraph 默认的 8000 端口
    env_file:
      - .env                                      # 载入环境变量配置文件
    environment:
      # 覆盖或注入 Docker 网络内的连接地址（容器间通过服务名直接通信）
      - DATABASE_URI=postgres://postgres:123456@postgres:5432/langgraph
      - REDIS_URI=redis://redis:6379
    depends_on:
      - postgres                                  # 确保 postgres 启动后再启动 agent
      - redis                                     # 确保 redis 启动后再启动 agent
```

**运行命令**
```shell
# 启动所有容器（后台运行）
docker compose up -d

# 查看运行状态
docker compose ps

# 查看 Agent 运行日志
docker compose logs -f agent
```