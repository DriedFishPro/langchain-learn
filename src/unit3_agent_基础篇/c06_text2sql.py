import os
import sqlite3
from langchain.agents import create_agent
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from util.print_util import print_event_stream

# 1. 动态获取当前文件 (text2sql_agent.py) 所在的绝对目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 拼接出 db 文件的绝对路径
DB_PATH = os.path.join(CURRENT_DIR, "sqlite", "text2sql.db")

def get_db_schema() -> str:
    """获取数据库中所有表的结构（DDL定义）。在编写任何 SQL 查询之前，必须先调用此工具来了解表名和列名。"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 查询所有用户创建的表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_info = []
    for table in tables:
        table_name = table[0]
        if table_name == 'sqlite_sequence':  # 排除内置表
            continue
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        ddl = cursor.fetchone()[0]
        schema_info.append(ddl)

    conn.close()
    return "\n\n".join(schema_info)


def execute_sql(query: str) -> str:
    """执行标准 SQL 查询语句，并以字符串形式返回执行结果。如果 SQL 语法有误，将返回具体的错误信息。"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # 执行 LLM 生成的 SQL
        cursor.execute(query)
        # 判断是否为查询语句
        if query.strip().upper().startswith('SELECT'):
            result = cursor.fetchall()
            # 提取列名，让结果可读性更好
            columns = [description[0] for description in cursor.description]
            formatted_result = [f"Columns: {columns}"]
            for row in result:
                formatted_result.append(str(row))

            return "\n".join(formatted_result)
        else:
            # 修改操作（UPDATE/INSERT/DELETE）返回影响行数
            conn.commit()
            return f"操作成功，影响了 {cursor.rowcount} 行数据"
    except Exception as e:
        # 关键点：把异常原封不动抛给 LLM，它看到报错后会自动修正 SQL 并重新运行！
        return f"SQL Error: {str(e)}"
    finally:
        conn.close()


def get_text2sql_prompt() -> str:
    return (
        "你是一个精通 SQL 的数据库分析专家。"
        "当你收到用户的数据查询需求时，请先调用 get_db_schema 工具了解表结构，"
        "然后编写正确的 SQLite 语句并调用 execute_sql 得到结果。"
        "如果执行报错，请仔细检查错误信息并修正 SQL，直到获得正确答案。"
        "最后，用通俗易懂的中文总结数据并回答用户。"
    )


if __name__ == "__main__":
    model = create_model(enable_thinking=True)

    # 组装 Agent，把我们的数据库的两个工具塞进去
    agent = create_agent(
        model=model,
        tools=[get_db_schema, execute_sql],
        system_prompt=get_text2sql_prompt()
    )

    print("=== 数据库查询助手已启动 ===")
    print("请输入你的问题：\n")  # 查询所有员工数据

    while True:
        user_input = input("你: ").strip()
        messages = {"messages": [{"role": "user", "content": user_input}]}
        stream = agent.stream_events(input=messages, version="v3")
        print_event_stream(stream)