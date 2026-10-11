import json
import sqlite3
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool
from unit1_大模型基础篇.a03_langchain调用模型 import create_model


class SqlAgent:

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        conn = sqlite3.connect(str(self.base_dir / 'db' / 'sqlite' / 'insurance.db'), check_same_thread=False)
        self.cursor = conn.cursor()
        self.model = create_model()
        system_prompt = (self.base_dir / "prompt" / "sql_prompt.md").read_text(encoding="utf-8")

        # 获取表结构，注入到提示词中
        db_schema = self.get_db_schema()
        system_prompt = system_prompt.replace('{db_schema}', db_schema)
        # print(system_prompt)

        self.agent = create_agent(
            model=self.model,
            system_prompt=system_prompt,
            tools=[tool(self.execute_sql, parse_docstring=True)],
        )

    def get_db_schema(self) -> str:
        """获取数据库中所有表的结构（DDL定义）"""
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' and name not in ('checkpoints', 'writes')")
        tables = self.cursor.fetchall()
        schema_list = []

        for table in tables:
            table_name = table[0]
            self.cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            ddl = self.cursor.fetchone()[0]

            self.cursor.execute(f"SELECT * FROM {table_name} limit 3;")
            # 1. 获取列名：cursor.description 是一个元组列表，其中第一个元素就是列名
            columns = [col[0] for col in self.cursor.description]
            # 2. 获取数据行
            rows = self.cursor.fetchall()
            # 3. 将每一行数据和列名打包成字典
            examples = [dict(zip(columns, row)) for row in rows]

            schema = {'table': table_name, 'ddl': ddl, 'examples': examples}
            schema_list.append(schema)

        return json.dumps(schema_list, ensure_ascii=False)

    def execute_sql(self, sql: str) -> str:
        """
        执行标准 SQL 查询语句，并以字符串形式返回执行结果

        Args:
            sql：要执行的sql语句
        """
        try:
            print(f"[sql agent]：执行SQL：{sql}")
            # 判断是否为查询语句
            if sql.strip().upper().startswith('SELECT'):
                self.cursor.execute(sql)
                result = self.cursor.fetchall()

                # 如果查询结果为空，直接提示
                if not result and not self.cursor.description:
                    return "查询成功，但未返回任何数据。"

                # 提取列名
                columns = [description[0] for description in self.cursor.description]
                # 将每一行数据和列名打包成字典列表
                data_list = [dict(zip(columns, row)) for row in result]

                return json.dumps(data_list, ensure_ascii=False, default=str)
            else:
                return "SQL执行失败：只允许执行select查询语句"
        except Exception as e:
            # 关键点：把异常原封不动抛给 LLM，它看到报错后会自动修正 SQL 并重新运行！
            return f"execute_sql 执行失败: {str(e)}"

    def run(self, input: str):
        response = self.agent.invoke(input={"messages": [("user", input)]})
        print('-' * 80 + "Sql Agent 消息列表↓↓↓" + '-' * 80)

        for message in response["messages"]:
            print(f'[sql agent]：{type(message)}，{message}', end="\n\n")

        print('-' * 80 + "Sql Agent 消息列表↑↑↑" + '-' * 80, end="\n\n")
        return response["messages"][-1].content


if __name__ == "__main__":
    sql_agent = SqlAgent()
    # result = sql_agent.get_db_schema()
    # result = sql_agent.execute_sql("select * from products")

    while True:
        query = input('请输入想要检索的内容：')
        if not query: continue
        result = sql_agent.run(query)
        print(f"最终回答：{result}")
        print('=' * 100)