import os
import asyncio
from dotenv import load_dotenv

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = ChatTongyi(
    model=os.getenv("LLM_MODEL_ID", "qwen-max"),
    dashscope_api_key=os.getenv("LLM_API_KEY"),
    temperature=0.7
)

# 1. 定义两个独立的 Prompt
prompt_loan = PromptTemplate.from_template("作为贷款专家，分析以下问题中的贷款需求：{query}")
prompt_credit = PromptTemplate.from_template("作为信用卡专家，分析以下问题中的信用卡需求：{query}")

chain_loan = prompt_loan | llm | StrOutputParser()
chain_credit = prompt_credit | llm | StrOutputParser()

# 2. 构建并行执行节点 (RunnableParallel)
# 这里的 chain_loan 和 chain_credit 会【同时】向通义千问发起请求
parallel_chain = RunnableParallel(
    loan_analysis=chain_loan,
    credit_analysis=chain_credit
)

# 3. 汇总节点 (接收上面两个并行节点合并后的字典)
summary_prompt = PromptTemplate.from_template(
    "结合以下两方面的分析结果，为客户生成一份综合金融服务方案：\n"
    "1. 贷款分析：{loan_analysis}\n"
    "2. 信用卡分析：{credit_analysis}"
)
summary_chain = summary_prompt | llm | StrOutputParser()

# 4. 组装总链：先【并行】，后【串行汇总】
full_chain = parallel_chain | summary_chain

# 5. 测试运行
async def main():
    print("🚀 开始并行请求通义千问 API...")
    # 这里的 loan 和 credit 请求是同时发出的，整体耗时取决于两者中较慢的那一个
    res = await full_chain.ainvoke({"query": "我想借款 10 万块买车，顺便办一张适合消费刷卡的信用卡"})
    print("\n👉 最终综合响应：\n", res)

if __name__ == "__main__":
    asyncio.run(main())