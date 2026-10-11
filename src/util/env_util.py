import os

def setup_apikey():
    os.environ["DASHSCOPE_API_KEY"] = "sk-ws-H.EPHYMDL.vh6A.MEYCIQCUWuNafJ4C30cvZDbvY6NX4p8qVQ2o5o61P16FQ8kJKQIhAIl3w3PXXkpqh1NZcZgj9FEvIZU5GUKq76E69oyIovcx"
    os.environ["DASHSCOPE_WORKSPACE_ID"] = "ws-kbaeolgqzl8jnnkd"
    os.environ["OPENAI_API_KEY"] = "sk-tVgfqJD1eI0lUL9mAa0RsCjZgjdXX9f3FhT1uA8IrDQ4A2Uw"
    os.environ["GAO_DE_API_KEY"] = "558ed369729d83b30b44d0aae2ee203a"


def setup_fiddler():
    print("已开启fiddler代理")
    # 1. 让程序走 Fiddler 代理
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:8888"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8888"
    # 2. 指定信任 Fiddler 的 pem 证书（请替换为你电脑上的绝对路径）
    os.environ["SSL_CERT_FILE"] = r"C:\Users\njyu\Desktop\fiddler.pem"
    # 某些底层库也会读取这个变量，顺便一起设上
    os.environ["REQUESTS_CA_BUNDLE"] = r"C:\Users\njyu\Desktop\fiddler.pem"
    os.environ["CURL_CA_BUNDLE"] = r"C:\Users\njyu\Desktop\fiddler.pem"


def setup_langsmith():
    print("已开启langsmith追踪")
    os.environ['LANGSMITH_TRACING'] = 'true'
    os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGSMITH_API_KEY'] = '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'
    os.environ['LANGSMITH_PROJECT'] = 'langchain-learn'