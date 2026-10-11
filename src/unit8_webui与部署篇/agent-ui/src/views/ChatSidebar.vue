<template>
  <button class="new-chat-btn" @click="handleNewChat">+ 新建对话</button>

  <div class="history-list">
     <div class="history-item" :class="{ active: item.thread_id === chatStore.currentThreadId }"
      @click="handleSelectThread(item.thread_id)" v-for="item in sessionList" :key="item.thread_id">
      <span class="history-item-text">{{ item.values?.messages?.[0]?.content || 'New Chat' }}</span>
      <span class="delete-btn" @click.stop="handleDeleteThread(item.thread_id)">✕</span>
    </div>
  </div>
</template>

<script>
import { useChatStore } from '@/stores/store'
import { del, post } from '@/utils/request'

export default {
  name: 'ChatSidebar',
  components: { },

  data() {
    return {
      sessionList: [
        // {
        //   "thread_id":"019f8e48-5e2e-70bc-ae56-8be0b4d43909",
        //   "status":"idle",
        //   "metadata":{
        //     "graph_id":"agent",
        //     "assistant_id":"fe096781-5601-53d2-b2f6-0d3403f7e9ca"
        //   },
        //   "config":{
        //     "configurable":{ }
        //   },
        //   "created_at":"2026-07-23T09:22:15.533760+00:00",
        //   "updated_at":"2026-07-23T09:22:17.181161+00:00",
        //   "values":{
        //     "messages":[
        //       {
        //         "content":"对话测试",
        //         "additional_kwargs":{ },
        //         "response_metadata":{ },
        //         "type":"human",
        //         "name":null,
        //         "id":"019f8e48-5e2f-734a-8767-16227a77d23c"
        //       },
        //       {
        //         "content":[
        //           {
        //             "type":"text",
        //             "text":"您好！我是您的保险客服总控智能助手。请问有什么可以帮您的吗？",
        //             "index":0
        //           }
        //         ],
        //         "additional_kwargs":{ },
        //         "response_metadata":{
        //           "model_provider":"openai",
        //           "finish_reason":"stop",
        //           "model_name":"qwen3.5-plus-2026-04-20",
        //           "output_version":"v1"
        //         },
        //         "type":"ai",
        //         "name":null,
        //         "id":"lc_run--019f8e48-62b5-7662-9152-de165277bf91",
        //         "tool_calls":[ ],
        //         "invalid_tool_calls":[ ],
        //         "usage_metadata":{
        //           "input_tokens":3734,
        //           "output_tokens":18,
        //           "total_tokens":3752
        //         }
        //       }
        //     ]
        //   },
        //   "state_updated_at":"2026-07-23T09:22:17.181161+00:00",
        //   "interrupts":{ },
        //   "error":null
        // }
      ],
    }
  },

  computed: {
    chatStore() {
      return useChatStore()
    },
  },

   watch: {
    'chatStore.refreshFlag'() {
      this.getSessionList()
    },
  },

  async created() {
    await this.getSessionList()
  },

  methods: {
    // 加载会话列表
    async getSessionList(){
      this.sessionList = await post('/threads/search', {})
    },

    // 新建会话
    handleNewChat() {
      const uuid = crypto.randomUUID()
      console.log('handleNewChat', uuid)
      this.chatStore.setThreadId(uuid)
    },

    // 会话选中事件（切换会话）
    handleSelectThread(threadId) {
      this.chatStore.setThreadId(threadId)
    },

    // 删除会话
    async handleDeleteThread(threadId) {
      await del(`/threads/${threadId}`)
      await this.getSessionList()
    },
  },
}
</script>

<style scoped>

.new-chat-btn {
  background: transparent;
  border: 1px solid #565869;
  color: #fff;
  padding: 12px;
  border-radius: 6px;
  cursor: pointer;
  margin-bottom: 20px;
  text-align: left;
  transition: background 0.2s;
}
.new-chat-btn:hover { background: #2a2b32; }

.history-item {
  padding: 12px;
  color: #ececec;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  margin-bottom: 4px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.history-item:hover, .history-item.active {
  background: #343541;
}

.history-item-text {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}

.delete-btn {
  display: none;
  margin-left: 8px;
  color: #999;
  font-size: 12px;
  flex-shrink: 0;
}
.delete-btn:hover {
  color: #ff4d4f;
}
.history-item:hover .delete-btn {
  display: inline;
}
</style>
