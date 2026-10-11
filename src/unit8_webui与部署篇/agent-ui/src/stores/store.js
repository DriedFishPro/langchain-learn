import { ref } from 'vue'
import { defineStore } from 'pinia'

export const API_URL = 'http://localhost:2024'

export const useChatStore = defineStore('chat', () => {
  const currentThreadId = ref(null)
  const refreshFlag = ref(0)

  function setThreadId(id) {
    currentThreadId.value = id
  }

   function triggerRefresh() {
    refreshFlag.value++
  }

  return { currentThreadId, setThreadId, refreshFlag, triggerRefresh }
})
