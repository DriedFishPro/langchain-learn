import os
import requests
import rich

# 这里一定要记得配置一下环境变量
from util.env_util import setup_apikey
setup_apikey()

# 1. 配置请求的 URL（根据实际地域修改）
# url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" # 旧版
url = "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions" # 新版
url = url.replace("{WorkspaceId}", os.getenv("DASHSCOPE_WORKSPACE_ID"))

# 2. API Key
api_key = os.getenv("DASHSCOPE_API_KEY")
print(f'api_key={api_key}')

# 3. 设置 HTTP 请求头 (Headers)
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 4. 设置请求体数据 (JSON Data)
data = {
    "model": "qwen3.7-flash",  # 提示：实际使用时请确认该模型名称是否准确，如 qwen-plus
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请做个自我介绍"}
    ]
}

# 5. 发送 POST 请求
response = requests.post(url, headers=headers, json=data)
rich.print(response.json())