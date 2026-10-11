from typing import TypedDict
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class State(TypedDict):
    order_id: str # 订单id
    amount: float  # 订单金额，用于控制支付是否成功
    order_status: str # 订单状态：paid-已支付，closed-已关闭
    close_reason: str   # 订单关闭原因

# 模拟支付节点：根据订单金额动态决定成功/失败
def payment_node(state):
    amount = state.get("amount", 0)
    print(f"[payment_node] 尝试为订单 {state.get('order_id')} 扣款，金额={amount}...")

    # 业务规则：假设用户的账户余额是1000，当订单支付金额超过1000时，支付网关返回支付失败
    user_balance = 1000

    if amount > user_balance:
        print(f"[payment_node] 支付失败：余额不足！")
        raise Exception(f"支付失败：余额不足")

    print(f"[payment_node] 支付成功！")
    return {"order_status": "paid"}

# 最终节点
def final_node(state):
    print(f"[final_node] 订单流程结束，order_status={state.get('order_status')}")
    return {}

# 支付节点的异常处理：直接在此完成订单关闭逻辑
def payment_error_handler(state: State, error: NodeError) -> Command:
    print(f"[payment_error_handler] 支付失败：{error}，关闭订单")

    return Command(
        # 直接更新为最终的“关闭”状态和原因
        update={"order_status": "closed", "close_reason": f"{error.error}"},
        goto="final_node",  # 直接跳转到最终收尾节点
    )

if __name__ == "__main__":
    graph = (
        StateGraph(State)
        .add_node("payment_node", payment_node, error_handler=payment_error_handler)
        .add_node("final_node", final_node)

        .add_edge(START, "payment_node")
        .add_edge("payment_node", "final_node")
        .add_edge("final_node", END)
        .compile()
    )

    # 场景1：正常支付（金额较小，支付成功）
    # result1 = graph.invoke({"order_id": "ORD-001", "amount": 200, "order_status": "pending"})
    # print(f"最终结果1: {result1}")

    # 场景2：支付失败（余额不足，触发补偿：关闭订单）
    result2 = graph.invoke({"order_id": "ORD-002", "amount": 2000, "order_status": "pending"})
    print(f"最终结果2: {result2}")