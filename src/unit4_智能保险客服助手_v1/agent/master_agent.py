import sqlite3
from pathlib import Path
from uuid import uuid4
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from unit1_大模型基础篇.a03_langchain调用模型 import create_model
from unit4_智能保险客服助手_v1.agent.rag_agent import RagAgent
from unit4_智能保险客服助手_v1.agent.sql_agent import SqlAgent


class MasterAgent:
    def __init__(self):
        print('[MasterAgent] 实例化')
        # 实例化子 Agent（此时完成了数据库连接等操作，且只执行一次）
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model = create_model(enable_thinking=True)
        self.sub_rag_instance = RagAgent()
        self.sub_sql_instance = SqlAgent()
        self.tools = [
            tool(self.insurance_knowledge_expert, parse_docstring=True),
            tool(self.insurance_data_expert, parse_docstring=True)
        ]


    def insurance_knowledge_expert(self, query: str, product: str) -> str:
        """
        单文档知识检索: 当用户询问关于保险产品条款、理赔流程、保险业务咨询等专业问题时，调用此工具。

        Args:
            query: 用户咨询的问题
            product: 保险产品全称，如：长生爱康保重大疾病保险
        """
        print(f"[Master 检索] query: {query}，product: {product}")
        # 调用子 Agent 并返回纯文本
        rag_res = self.sub_rag_instance.run(input=query, source=product)
        return rag_res

    def insurance_data_expert(self, query: str) -> str:
        """
        结构化数据与运营统计查询，当用户询问关于保险业务数据时，调用此工具

        Args:
            query: 用户的数据查询诉求，必须是明确的中文业务描述
        """
        print(f"[Master 数据查询] query: {query}")
        # 调用子 SQL Agent，它内部会自己看 Schema、写 SQL 并执行返回 Markdown 表格
        sql_res = self.sub_sql_instance.run(input=query)
        return sql_res


    # 3. 构建 Master Agent
    def build_agent(self):
        print('[MasterAgent] build_agent')
        system_prompt = (self.base_dir / "prompt" / "master_prompt.md").read_text(encoding="utf-8")
        products = self.sub_sql_instance.execute_sql(sql="select product_name, aliases from products")
        system_prompt = system_prompt.replace('{products}', products)

        conn = sqlite3.connect(f"{self.base_dir}/db/sqlite/insurance.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()

        master_agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer
        )
        return master_agent


# 4. 运行测试
if __name__ == "__main__":
    master_agent_build = MasterAgent()
    master_agent = master_agent_build.build_agent()

    config = {"configurable": {"thread_id": str(uuid4())}}
    print('智能保险客服问答助手已启动，请输入你想咨询的问题')

    while True:
        user_input = input("你：")
        if user_input == "exit":
            break

        response = master_agent.invoke(input={"messages": [("user", user_input)]}, config=config)

        print('-'*80 + "Master Agent 消息列表↓↓↓" + '-'*80)
        for message in response["messages"]:
            print(type(message), message, end="\n\n")
        print('-' * 80 + "Master Agent 消息列表↑↑↑" + '-' * 80, end="\n\n")

        print(f'Master Agent 最终回复：\n{response["messages"][-1].content}', end="\n\n")