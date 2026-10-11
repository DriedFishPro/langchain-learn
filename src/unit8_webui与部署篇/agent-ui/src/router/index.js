import { createRouter, createWebHistory } from 'vue-router'
import ChatLayout from '../views/chat_layout.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'chat_layout',
      component: ChatLayout,
    },
  ],
})

export default router
