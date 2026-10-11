<script setup>
import { computed } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

// 配置 marked：启用 GFM（表格、删除线、任务列表等）和换行转 <br>
marked.setOptions({ gfm: true, breaks: true })

const props = defineProps({
  content: {
    type: String,
    default: ''
  }
})

// 将 markdown 解析为 HTML，并经过 DOMPurify 净化防止 XSS
const html = computed(() => {
  if (!props.content) return ''
  const rawHtml = marked.parse(props.content)
  return DOMPurify.sanitize(rawHtml)
})
</script>

<template>
  <div class="markdown-content" v-html="html"></div>
</template>

<style scoped>
.markdown-content {
  line-height: 1.6;
  word-wrap: break-word;
}

.markdown-content :deep(h1),
.markdown-content :deep(h2),
.markdown-content :deep(h3) {
  margin: 0.6em 0 0.3em;
  font-weight: 600;
}

.markdown-content :deep(h1) { font-size: 1.3em; }
.markdown-content :deep(h2) { font-size: 1.2em; }
.markdown-content :deep(h3) { font-size: 1.1em; }

.markdown-content :deep(p) {
  margin: 0.4em 0;
}

.markdown-content :deep(ul),
.markdown-content :deep(ol) {
  margin: 0.4em 0;
  padding-left: 1.5em;
}

.markdown-content :deep(li) {
  margin: 0.2em 0;
}

.markdown-content :deep(code) {
  background: rgba(0, 0, 0, 0.06);
  padding: 0.15em 0.35em;
  border-radius: 3px;
  font-size: 0.9em;
  font-family: 'Consolas', 'Monaco', monospace;
}

.markdown-content :deep(pre) {
  background: rgba(0, 0, 0, 0.06);
  padding: 0.8em 1em;
  border-radius: 6px;
  overflow-x: auto;
  margin: 0.5em 0;
}

.markdown-content :deep(pre code) {
  background: none;
  padding: 0;
  font-size: 0.9em;
}

.markdown-content :deep(blockquote) {
  border-left: 3px solid rgba(0, 0, 0, 0.15);
  padding-left: 1em;
  margin: 0.5em 0;
  color: rgba(0, 0, 0, 0.6);
}

.markdown-content :deep(table) {
  border-collapse: collapse;
  margin: 0.5em 0;
  width: 100%;
}

.markdown-content :deep(th),
.markdown-content :deep(td) {
  border: 1px solid rgba(0, 0, 0, 0.15);
  padding: 0.4em 0.7em;
  text-align: left;
}

.markdown-content :deep(a) {
  color: #0066cc;
  text-decoration: none;
}

.markdown-content :deep(a:hover) {
  text-decoration: underline;
}
</style>
