# import os
# from dotenv import load_dotenv
#
# # 核心组件最新标准导入路径
# from langchain_community.chat_models import ChatTongyi
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
#
# # 1. 加载环境变量 (.env)
# load_dotenv()
#
# # 2. 初始化通义千问模型
# llm = ChatTongyi(
#     model=os.getenv("LLM_MODEL_ID", "qwen-max"),
#     dashscope_api_key=os.getenv("LLM_API_KEY"),
#     temperature=0.7
# )
#
# # 3. 定义任务模板
# template_1 = PromptTemplate.from_template("请解析以下用户请求：{query}")
# template_2 = PromptTemplate.from_template("根据解析结果：{result}，生成最终响应。")
#
# # 4. 使用 LCEL 构建顺序任务链
# # 技巧：通过 (lambda x: {"result": x}) 实现变量名衔接，传给下一个 template
# chain = (
#     template_1
#     | llm
#     | StrOutputParser()
#     | (lambda text: {"result": text})  # 将第一阶段输出对齐为 second prompt 的 input variable
#     | template_2
#     | llm
#     | StrOutputParser()
# )
#
# # 5. 执行任务链
# response = chain.invoke({"query": "申请个人消费贷款需要满足哪些条件？"})
#
# print("👉 最终响应结果：\n", response)

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
if not api_key:
    raise ValueError("未检测到 DASHSCOPE_API_KEY 或 LLM_API_KEY，请在 .env 文件中配置。")


class LoanQueryAnalysis(BaseModel):
    intent: str = Field(description="用户核心意图，例如：查询准入条件、额度利率、申请材料或办理流程")
    loan_type: str = Field(description="贷款类别，例如：个人消费贷款")
    key_dimensions: list[str] = Field(description="需核查或解答的关键维度列表")
    missing_user_info: list[str] = Field(description="若要给出更精准评估，还需向用户确认的信息项")


llm = ChatTongyi(
    model=os.getenv("LLM_MODEL_ID", "qwen-max"),
    dashscope_api_key=api_key,
    temperature=0.3,
)


def retrieve_loan_rules(analysis: LoanQueryAnalysis) -> str:
    rules_by_dimension = {
        "年龄与身份": "通常要求年满 18 周岁且具有完全民事行为能力，一般不超过法定退休年龄或特定上限（以具体银行产品为准）。",
        "信用记录": "个人征信记录良好，无严重逾期或当前未结清不良记录，近期查询次数不过多。",
        "收入与还款能力": "具备稳定的合法收入来源或偿债能力，月还款额通常需控制在合理收入负债比（DTI）范围内。",
        "资金用途": "贷款资金必须用于合法合规的个人日常消费（如装修、家电、教育、医疗等），严禁挪用于购房、炒股、理财或偿还违规债务。",
        "申请材料": "一般需提供有效身份证件、实名手机号、收入/工作证明、银行流水及用途说明等。",
    }

    selected_rules = []
    for dim in analysis.key_dimensions:
        for key, rule in rules_by_dimension.items():
            if key in dim or dim in key:
                selected_rules.append(f"- 【{key}】{rule}")

    if not selected_rules:
        selected_rules = [f"- {v}" for v in rules_by_dimension.values()]

    return "\n".join(dict.fromkeys(selected_rules))


analyze_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个金融业务需求拆解助手。请将用户的贷款咨询拆解为结构化要素，"
        "明确意图、贷款类型、需要回答的关键维度，以及给出针对性建议前可能缺少的用户信息。"
    ),
    ("human", "用户请求：{query}"),
])
structured_llm = llm.with_structured_output(LoanQueryAnalysis)
analyze_chain = analyze_prompt | structured_llm


def enrich_context(data: dict) -> dict:
    analysis: LoanQueryAnalysis = data["analysis"]
    return {
        "intent": analysis.intent,
        "loan_type": analysis.loan_type,
        "key_dimensions": analysis.key_dimensions,
        "missing_user_info": analysis.missing_user_info,
        "retrieved_rules": retrieve_loan_rules(analysis),
    }


context_builder = RunnablePassthrough.assign(**{
    "intent": RunnableLambda(lambda x: x["analysis"].intent),
    "loan_type": RunnableLambda(lambda x: x["analysis"].loan_type),
    "key_dimensions": RunnableLambda(lambda x: x["analysis"].key_dimensions),
    "missing_user_info": RunnableLambda(lambda x: x["analysis"].missing_user_info),
    "retrieved_rules": RunnableLambda(lambda x: retrieve_loan_rules(x["analysis"])),
})

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一位严谨的个人信贷顾问。请严格依据提供的【准入规则参考】回答用户问题，"
        "不得编造具体银行利率或审批承诺；最后给出补充确认项与合规提示。"
    ),
    (
        "human",
        "原始问题：{query}\n"
        "拆解结果：\n"
        "- 意图：{intent}\n"
        "- 贷款类型：{loan_type}\n"
        "- 关键维度：{key_dimensions}\n"
        "准入规则参考：\n{retrieved_rules}\n"
        "待确认信息：{missing_user_info}"
    ),
])
answer_chain = answer_prompt | llm | StrOutputParser()

task_decomposition_chain = (
    RunnablePassthrough.assign(analysis=analyze_chain)
    | context_builder
    | answer_chain
)

if __name__ == "__main__":
    user_query = "申请个人消费贷款需要满足哪些条件？"
    final_answer = task_decomposition_chain.invoke({"query": user_query})
    print("👉 最终响应结果：\n")
    print(final_answer)