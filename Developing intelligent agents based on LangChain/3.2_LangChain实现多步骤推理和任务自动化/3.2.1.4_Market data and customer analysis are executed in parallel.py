import os
import asyncio
from dotenv import load_dotenv

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 1. 加载环境变量
load_dotenv()

# 2. 初始化通义千问模型
llm = ChatTongyi(
    model=os.getenv("LLM_MODEL_ID", "qwen-max"),
    dashscope_api_key=os.getenv("LLM_API_KEY"),
    temperature=0.3
)

# 3. 定义市场数据分析链 (Market Data Chain)
market_prompt = PromptTemplate.from_template(
    "【市场分析】根据以下市场数据进行简要趋势分析：\n{market_data}"
)
market_data_chain = market_prompt | llm | StrOutputParser()

# 4. 定义客户分析链 (Client Analysis Chain)
client_prompt = PromptTemplate.from_template(
    "【客户画像】根据以下客户信息分析其投资偏好与风险承受能力：\n{client_data}"
)
client_analysis_chain = client_prompt | llm | StrOutputParser()

# 5. 构建 LCEL 异步并行任务链
# 市场数据链与客户分析链将并发执行，提升响应效率
parallel_analysis_chain = RunnableParallel({
    "market_analysis": market_data_chain,
    "client_analysis": client_analysis_chain
})

# 6. 汇总整合链（可选：将并行分析的结果二次汇总为一份终审报告）
summary_prompt = PromptTemplate.from_template(
    "【综合报告】结合以下市场分析与客户分析，给出投资顾问建议：\n\n"
    "--- 市场分析 --- \n{market_analysis}\n\n"
    "--- 客户分析 --- \n{client_analysis}"
)
full_workflow_chain = parallel_analysis_chain | summary_prompt | llm | StrOutputParser()


# 7. 异步执行主函数
async def run_async_pipeline():
    input_data = {
        "market_data": "美联储降息25个基点，港股科技板块大幅反弹，市场流动性改善。",
        "client_data": "客户张总，35岁，企业高管，风险偏好中等偏高，期望年化收益10%-15%。"
    }

    print("🚀 开始并发执行市场与客户分析...")

    # 方式 A：单独执行并发分析（返回字典形式结果）
    parallel_results = await parallel_analysis_chain.ainvoke(input_data)
    print("\n[单独并发分析结果]:")
    print(f"市场分析结果：\n{parallel_results['market_analysis']}\n")
    print(f"客户分析结果：\n{parallel_results['client_analysis']}\n")

    print("--------------------------------------------------")

    # 方式 B：执行完整工作流（并行分析 + 最终报告汇总）
    print("🚀 执行完整并发+汇总工作流...")
    final_report = await full_workflow_chain.ainvoke(input_data)
    print("\n[最终综合报告]:")
    print(final_report)


if __name__ == "__main__":
    asyncio.run(run_async_pipeline())
