<template>
  <header class="chat-header">
    <div class="brand">🤖 LangChain Agent</div>
    <div class="status">
      <span v-if="isLoading" class="status-running">🟡 思考中...</span>
      <span v-else class="status-idle">🟢 就绪</span>
    </div>
  </header>

  <div ref="messagesContainer" class="messages-area">
    <div v-if="messages.length === 0" class="empty-state">
      <h2>有什么可以帮你的？</h2>
      <p>我是你的 AI 助手，可以回答问题、编写代码、分析数据</p>
    </div>

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

    <div v-if="isLoading" class="msg-item ai">
      <div class="msg-avatar">🤖</div>
      <div class="msg-content typing">正在输出...</div>
    </div>
  </div>

  <footer class="input-area">
    <div v-if="error" class="error-msg">⚠️ {{ error }}</div>

    <div class="input-box">
      <textarea ref="inputEl" v-model="userInput" @keydown="handleKeydown" @input="autoResize"
        placeholder="输入消息，Enter 发送，Shift+Enter 换行" rows="1" :disabled="isLoading"
      ></textarea>
      <button class="send-btn" @click="sendMessage" :disabled="!userInput.trim() || isLoading">
        发送
      </button>
    </div>
    <div class="input-hint">内容由 AI 生成，仅供参考</div>
  </footer>
</template>

<script>
import { useStream } from '@langchain/vue'
import MarkdownRenderer from '@/components/MarkdownRenderer.vue'
import { useChatStore, API_URL } from '@/stores/store'

export default {
  name: 'ChatMain',
  components: { MarkdownRenderer },

  setup() {
    const chatStore = useChatStore()

    const stream = useStream({
      apiUrl: API_URL,
      assistantId: 'agent',
      threadId: chatStore.currentThreadId,
    })
    return { stream }
  },

  data() {
    return {
      userInput: '',
    }
  },

  computed: {
    messages() { return this.stream.messages.value },
    isLoading() { return this.stream.isLoading.value },
    error() { return this.stream.error.value },
  },

  watch: {
    messages: {
      handler() { this.scrollToBottom() },
      deep: true,
    },
  },

  methods: {
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

    handleKeydown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        this.sendMessage()
      }
    },

    autoResize() {
      const el = this.$refs.inputEl
      if (!el) return
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 200) + 'px'
    },

    resetTextareaHeight() {
      this.$nextTick(() => {
        if (this.$refs.inputEl) this.$refs.inputEl.style.height = 'auto'
      })
    },

    async scrollToBottom() {
      await this.$nextTick()
      if (this.$refs.messagesContainer) {
        this.$refs.messagesContainer.scrollTop = this.$refs.messagesContainer.scrollHeight
      }
    },

    getMessageText(content) {
      if (typeof content === 'string') return content
      if (Array.isArray(content)) {
        return content.filter(b => b.type === 'text').map(b => b.text).join('')
      }
      return ''
    },

    isHuman(msg) { return msg.type === 'human' },
    isAI(msg) { return msg.type === 'ai' },
  },
}
</script>

<style scoped>
.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  border-bottom: 1px solid #4d4f5c; /* 调暗边框颜色 */
  font-weight: 500;
  color: #ececec;
}

.messages-area {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
}

/* --- 消息流 --- */
.msg-item {
  display: flex;
  gap: 16px;
  max-width: 800px;
  margin: 0 auto 32px;
}

.msg-avatar {
  width: 32px;
  height: 32px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 18px;
}

/* 头像背景微调以适应深色 */
.human .msg-avatar { background: #5436DA; } /* 用户的给个深紫/蓝色 */
.ai .msg-avatar { background: #10a37f; }
.tool .msg-avatar { background: #444654; border: 1px solid #565869; } /* 工具的给个深灰 */

.msg-content {
  flex: 1;
  line-height: 1.6;
  color: #d1d5db; /* 聊天文本颜色 */
  min-width: 0;
}
.msg-content.typing { color: #9ca3af; }

.empty-state {
  text-align: center;
  margin-top: 15vh;
  color: #9ca3af;
}
.empty-state h2 { color: #ececec; }

/* --- 底部输入栏 --- */
.input-area {
  padding: 24px;
  background: #343541; /* 与主背景一致 */
}

.input-box {
  max-width: 800px;
  margin: 0 auto;
  display: flex;
  align-items: flex-end;
  gap: 12px;
  background: #40414f; /* 输入框背景设为稍亮的深灰 */
  border: 1px solid #565869;
  border-radius: 12px;
  padding: 12px;
}
.input-box:focus-within { border-color: #10a37f; }

.input-box textarea {
  flex: 1;
  border: none;
  background: transparent;
  outline: none;
  resize: none;
  max-height: 200px;
  color: #ececec; /* 输入的文字颜色 */
  font-family: inherit;
  font-size: 15px;
  line-height: 1.5;
}
.input-box textarea::placeholder {
  color: #8e8ea0;
}

.send-btn {
  padding: 8px 16px;
  background: #10a37f;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  height: 36px;
}
.send-btn:disabled {
  background: #40414f; /* 禁用状态的按钮颜色 */
  color: #565869;
  cursor: not-allowed;
}

.input-hint {
  text-align: center;
  font-size: 12px;
  color: #8e8ea0;
  margin-top: 12px;
}

.error-msg {
  max-width: 800px;
  margin: 0 auto 12px;
  color: #ef4444; /* 深色模式下的红色亮一点 */
  text-align: center;
  font-size: 14px;
}
</style>
