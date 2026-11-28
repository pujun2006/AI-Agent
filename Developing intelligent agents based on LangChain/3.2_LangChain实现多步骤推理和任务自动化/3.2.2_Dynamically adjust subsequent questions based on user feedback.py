import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
if not api_key:
    raise ValueError("未检测到 DASHSCOPE_API_KEY 或 LLM_API_KEY，请在 .env 文件中配置。")

llm = ChatTongyi(
    model=os.getenv("LLM_MODEL_ID", "qwen-max"),
    dashscope_api_key=api_key,
    temperature=0.3,
)


# 1. 定义动态决策契约：评估当前信息完整度与下一步追问方向
class FollowUpDecision(BaseModel):
    is_complete: bool = Field(
        description="判断当前收集到的信息是否已足以给出清晰的初步方案/建议；若仍缺关键信息或用户反馈含糊，则为 False"
    )
    known_facts: list[str] = Field(
        description="截至目前从对话中已确认的关键事实（如金额、用途、职业/收入、期限、征信或偏好）"
    )
    missing_or_unclear_points: list[str] = Field(
        description="当前缺失或需要进一步澄清的要点"
    )
    next_question: str = Field(
        description="若 is_complete 为 False，生成一句自然、有针对性的下一步追问；若为 True，可留空或写确认语"
    )
    summary_or_advice: str = Field(
        description="若 is_complete 为 True，输出基于已知信息的初步方案与合规提示；否则可简述当前理解"
    )


decision_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一位专业、耐心的信贷需求顾问。请结合历史对话与最新用户反馈，"
        "动态判断信息是否充分并决定下一步询问：\n"
        "1. 每次只聚焦 1~2 个最关键的缺失点或模糊反馈发问，避免一次性抛出一长串问题给用户造成负担。\n"
        "2. 如果用户反馈表示“不确定/大概/先看情况”，请顺着用户反馈降低追问门槛，给出常见选项或折中问法。\n"
        "3. 不得编造具体银行利率或审批承诺，涉及资金用途需提示合法合规。"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "用户最新反馈：{user_feedback}"),
])

structured_decision_llm = llm.with_structured_output(FollowUpDecision)
decision_chain = decision_prompt | structured_decision_llm


# 2. 封装多轮动态问询控制器
class DynamicInquirySession:
    def __init__(self, max_turns: int = 5):
        self.chat_history: list = []
        self.max_turns = max_turns
        self.turn_count = 0
        self.latest_decision: FollowUpDecision | None = None

    def step(self, user_feedback: str) -> str:
        self.turn_count += 1

        decision: FollowUpDecision = decision_chain.invoke({
            "chat_history": self.chat_history,
            "user_feedback": user_feedback,
        })
        self.latest_decision = decision

        # 记录本轮对话历史
        self.chat_history.append(HumanMessage(content=user_feedback))

        # 达到最大追问轮次时，强制收敛输出当前阶段的建议
        if decision.is_complete or self.turn_count >= self.max_turns:
            final_text = self._build_final_response(decision)
            self.chat_history.append(AIMessage(content=final_text))
            return final_text

        assistant_reply = (
            f"（已确认：{', '.join(decision.known_facts) if decision.known_facts else '初步需求'}）\n"
            f"{decision.next_question}"
        )
        self.chat_history.append(AIMessage(content=assistant_reply))
        return assistant_reply

    def _build_final_response(self, decision: FollowUpDecision) -> str:
        if decision.summary_or_advice.strip():
            return decision.summary_or_advice

        # 兜底生成总结
        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", "请根据以下已确认信息和对话历史，给出清晰的初步方案建议与合规提示，不编造具体利率。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "已确认要点：{known_facts}"),
        ])
        fallback_chain = fallback_prompt | llm | StrOutputParser()
        return fallback_chain.invoke({
            "chat_history": self.chat_history,
            "known_facts": "；".join(decision.known_facts) or "基础消费贷款意向",
        })


# 3. 模拟多轮交互（实际接入 Web/CLI 时，改为循环读取 input() 或前端消息）
if __name__ == "__main__":
    session = DynamicInquirySession(max_turns=4)

    simulated_user_inputs = [
        "我想申请一笔个人消费贷款，但不知道怎么选",
        "大概 8 万左右吧，主要是家里装修，不过具体预算还没完全定",
        "我是上班族，月薪税后大概 1.5 万，社保正常缴纳，想分 12 到 24 期",
    ]

    for idx, msg in enumerate(simulated_user_inputs, start=1):
        print(f"\n--- 第 {idx} 轮 ---")
        print(f"用户：{msg}")
        reply = session.step(msg)
        print(f"助手：\n{reply}")

        if session.latest_decision and session.latest_decision.is_complete:
            print("\n✅ 关键信息已收集齐备，结束动态追问。")
            break