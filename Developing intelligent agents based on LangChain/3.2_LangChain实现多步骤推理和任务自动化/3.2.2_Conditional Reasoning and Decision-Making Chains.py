import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
if not api_key:
    raise ValueError("未检测到 DASHSCOPE_API_KEY 或 LLM_API_KEY，请在 .env 文件中配置。")

llm = ChatTongyi(
    model=os.getenv("LLM_MODEL_ID", "qwen-max"),
    dashscope_api_key=api_key,
    temperature=0.2,
)


# 1. 定义全局共享状态（决策链条的“工作记忆”）
class DecisionState(TypedDict):
    user_query: str
    search_query: str
    intent: Literal["loan_consult", "complaint", "general_chat"]
    retrieved_docs: str
    is_sufficient: bool
    retry_count: int
    final_answer: str


# 2. 结构化推理契约
class IntentDecision(BaseModel):
    intent: Literal["loan_consult", "complaint", "general_chat"] = Field(
        description="判断用户意图：loan_consult(贷款业务咨询)、complaint(投诉/异常反馈)、general_chat(闲聊/其他)"
    )
    normalized_query: str = Field(description="提取并标准化后的核心检索词")


class RetrievalGrade(BaseModel):
    is_sufficient: bool = Field(description="判断检索到的资料是否足以准确回答用户问题")
    rewritten_query: str = Field(description="若不充分，给出更精准的改写检索词；充分则保持原样")


# 3. 定义各个决策与执行节点
def classify_intent_node(state: DecisionState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "分析用户请求意图，并提炼适合知识库检索的标准化关键词。"),
        ("human", "{query}"),
    ])
    result: IntentDecision = (prompt | llm.with_structured_output(IntentDecision)).invoke(
        {"query": state["user_query"]}
    )
    return {
        "intent": result.intent,
        "search_query": result.normalized_query,
        "retry_count": 0,
    }


def retrieve_docs_node(state: DecisionState) -> dict:
    """模拟动态环境下的知识库检索：不同检索词命中的内容不同。"""
    q = state["search_query"]
    if "消费" in q or "装修" in q or "准入" in q:
        docs = "个人消费贷款准入要点：年满18周岁、征信良好、具备稳定合法收入，资金仅限合法日常消费，严禁流入楼市或股市。"
    else:
        docs = "未检索到具体信贷产品细则，仅命中通用客服引导页。"
    return {"retrieved_docs": docs}


def grade_retrieval_node(state: DecisionState) -> dict:
    """条件评估节点：判断当前检索结果是否满足回答要求，否则改写检索词准备重试。"""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "评估【检索内容】是否足以回答【用户问题】。如果内容过于宽泛或未命中具体规则，"
            "请将 is_sufficient 设为 False，并将检索词改写为包含“个人消费贷款 准入 条件”等更明确业务词的查询。"
        ),
        ("human", "用户问题：{query}\n当前检索词：{search_query}\n检索内容：{docs}"),
    ])
    grade: RetrievalGrade = (prompt | llm.with_structured_output(RetrievalGrade)).invoke({
        "query": state["user_query"],
        "search_query": state["search_query"],
        "docs": state["retrieved_docs"],
    })
    return {
        "is_sufficient": grade.is_sufficient,
        "search_query": grade.rewritten_query,
        "retry_count": state["retry_count"] + 1,
    }


def generate_loan_answer_node(state: DecisionState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "根据检索资料严谨回答问题，不编造利率或审批承诺；若资料有限请如实说明。"),
        ("human", "问题：{query}\n参考资料：{docs}"),
    ])
    ans = (prompt | llm | StrOutputParser()).invoke({
        "query": state["user_query"],
        "docs": state["retrieved_docs"],
    })
    return {"final_answer": ans}


def handle_complaint_node(state: DecisionState) -> dict:
    return {"final_answer": "已为您识别为异常/投诉类反馈，正在转接人工服务专员并优先核查处理。"}


def handle_general_node(state: DecisionState) -> dict:
    ans = llm.invoke(f"请礼貌、简洁地回应用户：{state['user_query']}").content
    return {"final_answer": ans}


# 4. 定义条件路由函数（动态路径选择的核心）
def route_by_intent(state: DecisionState) -> str:
    if state["intent"] == "loan_consult":
        return "retrieve_docs"
    if state["intent"] == "complaint":
        return "handle_complaint"
    return "handle_general"


def route_after_grading(state: DecisionState) -> str:
    # 信息充分，或已达到最大重试次数（最多重检 2 次），进入最终生成
    if state["is_sufficient"] or state["retry_count"] >= 2:
        return "generate_loan_answer"
    # 信息不足且未超限，回退到检索节点使用改写后的查询词重试
    return "retrieve_docs"


# 5. 组装条件状态图
workflow = StateGraph(DecisionState)
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("retrieve_docs", retrieve_docs_node)
workflow.add_node("grade_retrieval", grade_retrieval_node)
workflow.add_node("generate_loan_answer", generate_loan_answer_node)
workflow.add_node("handle_complaint", handle_complaint_node)
workflow.add_node("handle_general", handle_general_node)

workflow.add_edge(START, "classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_docs": "retrieve_docs",
        "handle_complaint": "handle_complaint",
        "handle_general": "handle_general",
    },
)
workflow.add_edge("retrieve_docs", "grade_retrieval")
workflow.add_conditional_edges(
    "grade_retrieval",
    route_after_grading,
    {
        "retrieve_docs": "retrieve_docs",
        "generate_loan_answer": "generate_loan_answer",
    },
)
workflow.add_edge("generate_loan_answer", END)
workflow.add_edge("handle_complaint", END)
workflow.add_edge("handle_general", END)

app = workflow.compile()

if __name__ == "__main__":
    # 测试：初始表述较模糊，触发“首次检索不足 -> 自动改写检索词 -> 二次检索命中 -> 生成回复”
    result = app.invoke({"user_query": "我想借点钱搞家里翻新，有啥门槛？"})
    print(f"识别意图：{result['intent']} | 检索次数：{result['retry_count']}")
    print("最终回复：\n", result["final_answer"])