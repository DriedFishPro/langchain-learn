from functools import lru_cache
from unit4_智能保险客服助手_v1.agent.master_agent import MasterAgent
from unit6_智能保险客服助手_v2.graph.master_graph import MasterGraph


# "agent": "./agent_server.py:build_agent",
@lru_cache(maxsize=1)
def build_agent():
    agent_builder = MasterAgent()
    return agent_builder.build_agent()

# "graph": "./agent_server.py:build_graph"
@lru_cache(maxsize=1)
def build_graph():
    graph_builder = MasterGraph()
    return graph_builder.build_graph()