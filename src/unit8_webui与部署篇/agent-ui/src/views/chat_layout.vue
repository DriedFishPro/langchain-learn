<template>
  <div class="app-layout">
    <aside class="sidebar">
      <chat-sidebar></chat-sidebar>
    </aside>

    <main class="chat-main">
      <chat-main :key="chatStore.currentThreadId || 'new'"></chat-main>
    </main>
  </div>
</template>

<script>

import ChatSidebar from "@/views/ChatSidebar.vue";
import ChatMain from "@/views/ChatMain.vue";
import { useChatStore } from '@/stores/store'

export default {
  name: 'ChatLayout',
  components: {ChatSidebar, ChatMain},

  data() {
    return {

    }
  },

  computed: {
    chatStore() {
      return useChatStore()
    },
  },

  watch: {

  },

  methods: {

  }
}
</script>

<style scoped>
.app-layout {
  display: flex;
  height: 100vh;
  background: #343541; /* 整体底色调整为深色 */
  font-family: system-ui, -apple-system, sans-serif;
}

/* --- 左侧边栏 --- */
.sidebar {
  width: 260px;
  background: #202123;
  color: #fff;
  display: flex;
  flex-direction: column;
  padding: 12px;
  flex-shrink: 0;
}


/* --- 右侧主界面 (深色模式) --- */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #343541; /* 右侧主区域深色背景 */
  color: #ececec;      /* 全局文字设为浅色 */
  min-width: 0;
}
</style>
