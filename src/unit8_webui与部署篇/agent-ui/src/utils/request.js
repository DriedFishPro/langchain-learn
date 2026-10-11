import { API_URL } from '@/stores/store'

async function request(url, options = {}) {
  const response = await fetch(`${API_URL}${url}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }

  if (response.status === 204 || response.headers.get('content-length') === '0') {
    return null
  }

  const text = await response.text()
  return text ? JSON.parse(text) : null
}

export function get(url, params = {}) {
  const queryString = new URLSearchParams(params).toString()
  const fullUrl = queryString ? `${url}?${queryString}` : url
  return request(fullUrl, { method: 'GET' })
}

export function post(url, data = {}) {
  return request(url, {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export function del(url, data = {}) {
  return request(url, {
    method: 'DELETE',
    body: JSON.stringify(data),
  })
}
