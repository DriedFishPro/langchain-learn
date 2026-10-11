import json
import sqlite3
import rich
from pathlib import Path
from typing import TypedDict, List, Annotated, Literal
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from unit1_大模型基础篇.a03_langchain调用模型 import create_model


class SqlState(TypedDict):
    messages: Annotated[list, add_messages] # 对话历史
    db_schema: List[dict[str, any]]  # 表结构
    execute_success: bool  # SQL执行结果标志
    loop_count: int  # 重试次数


class SqlGraph:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model = create_model()
        self.conn = sqlite3.connect(self.base_dir / "db" / "sqlite" / "insurance.db", check_same_thread=False)
        self.cursor = self.conn.cursor()


    # 节点1：获取数据库 schema
    def get_schema_node(self, state: SqlState):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' and name not in ('checkpoints', 'writes')")
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

        return {"db_schema": schema_list, "loop_count": 0}

    def generate_sql_node(self, state: SqlState):
        print(f"[generate_sql_node] 开始生成sql")
        prompt_template = (self.base_dir / "prompt" / "sql_prompt.md").read_text(encoding="utf-8")
        prompt = prompt_template.replace("{db_schema}", json.dumps(state["db_schema"], ensure_ascii=False, default=str))

        # print(f"[generate_sql_node] prompt: {prompt}", end="\n\n")
        messages = [("system", prompt)] + state["messages"]
        ai_message = self.model.invoke(messages)
        return {"messages": [ai_message], "loop_count": state.get("loop_count", 0) + 1}

    def execute_sql_node(self, state: SqlState) -> str:
        try:
            sql = state["messages"][-1].content
            print(f"[execute_sql_node]：执行SQL：{sql}")

            # 判断是否为查询语句
            if sql.strip().upper().startswith('SELECT'):
                self.cursor.execute(sql)
                fetch_result = self.cursor.fetchall()

                if not fetch_result and not self.cursor.description:
                    result = {'content': '查询成功，但未返回任何数据', 'execute_success': True}
                else:
                    columns = [description[0] for description in self.cursor.description]
                    data_list = [dict(zip(columns, row)) for row in fetch_result]
                    result = {'content': json.dumps(data_list, ensure_ascii=False, default=str), 'execute_success': True}
            else:
                result = {'content': 'SQL执行失败：只允许执行select查询语句', 'execute_success': False}
        except Exception as e:
            # 把异常原封不动抛给 LLM，它看到报错后会自动修正 SQL 并重新运行！
            result = {'content': f"SQL执行失败: {str(e)}", 'execute_success': False}

        print(f"[execute_sql_node]：执行结果：{result}")
        return {"messages": [HumanMessage(content=result['content'])], "execute_success": result['execute_success']}

    def route_after_execute(self, state: SqlState) -> Literal["generate_sql", END]:
        print(f"[route_after_execute] 开始判断sql执行结果 execute_success={state['execute_success']}", end="\n\n")

        if not state.get("execute_success") and state["loop_count"] < 3:
            print(f"[SQL Graph] 触发自纠错 (第{state['loop_count']}/3次)")
            return "generate_sql"
        return END

    def build(self):
        builder = StateGraph(SqlState)

        builder.add_node("get_schema", self.get_schema_node)
        builder.add_node("generate_sql", self.generate_sql_node)
        builder.add_node("execute_sql", self.execute_sql_node)

        builder.add_edge(START, "get_schema")
        builder.add_edge("get_schema", "generate_sql")
        builder.add_edge("generate_sql", "execute_sql")
        builder.add_conditional_edges("execute_sql", self.route_after_execute)
        return builder.compile()

if __name__ == "__main__":
    sql_graph = SqlGraph()
    graph = sql_graph.build()

    while True:
        query = input('请输入想要检索的内容：')
        if not query: continue

        result = graph.invoke({"messages": [HumanMessage(content=query)]})
        rows = json.loads(result['messages'][-1].content)

        print(f"最终回答：")
        for row in rows: print(row)
        print('=' * 100, end="\n\n")