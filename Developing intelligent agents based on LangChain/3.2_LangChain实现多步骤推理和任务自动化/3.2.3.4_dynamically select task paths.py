import os
import asyncio
from dotenv import load_dotenv

import langchain
from langchain_community.cache import InMemoryCache
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

# 1. 加载环境变量并初始化全局 LLM 缓存
load_dotenv()
langchain.llm_cache = InMemoryCache()  # 开启 LangChain 内置 LLM 响应缓存

# 模拟数据库查询客户偏好的函数（带本地字典缓存兜底）
CLIENT_DB = {
    "CLI-001": {"name": "张总", "client_type": "短期投资", "risk_level": "高"},
    "CLI-002": {"name": "李女士", "client_type": "长期投资", "risk_level": "低"}
}


def query_client_preferences(client_id: str) -> dict:
    print(f"🔍 [数据库查询] 正在从 DB 读取客户 {client_id} 的偏好数据...")
    return CLIENT_DB.get(client_id, {"name": "未知客户", "client_type": "长期投资", "risk_level": "低"})


# 2. 初始化通义千问模型
llm = ChatTongyi(
    model=os.getenv("LLM_MODEL_ID", "qwen-max"),
    dashscope_api_key=os.getenv("LLM_API_KEY"),
    temperature=0.3
)

# 3. 定义不同投资策略的处理链
short_term_prompt = PromptTemplate.from_template(
    "【短期高频策略建议】针对客户 {name}（风险等级：{risk_level}）：\n"
    "请结合当前宏观市场与热点，制定一份 1-3 个月的短期对冲与套利策略方案。"
)
short_term_strategy_chain = short_term_prompt | llm | StrOutputParser()

long_term_prompt = PromptTemplate.from_template(
    "【长期价值配置建议】针对客户 {name}（风险等级：{risk_level}）：\n"
    "请结合资产配置模型，制定一份 3-5 年的稳健核心-卫星（Core-Satellite）资产配置方案。"
)
long_term_strategy_chain = long_term_prompt | llm | StrOutputParser()

default_chain = PromptTemplate.from_template(
    "【标准平衡策略】针对客户 {name}：提供标准稳健型资产配置建议。"
) | llm | StrOutputParser()

# 4. 使用 RunnableBranch 构建现代 LCEL 动态路由
# 语法结构：RunnableBranch((条件1, 链1), (条件2, 链2), ..., 默认链)
strategy_router_branch = RunnableBranch(
    (
        lambda x: x["client_type"] == "短期投资",
        short_term_strategy_chain
    ),
    (
        lambda x: x["client_type"] == "长期投资",
        long_term_strategy_chain
    ),
    default_chain
)


# 5. 组装完整工作流：数据获取/缓存 -> 动态路由分支
async def process_client_investment_plan(client_id: str):
    # 步骤 A：获取客户偏好信息（可在此添加业务级数据缓存）
    client_info = query_client_preferences(client_id)

    print(f"🎯 [动态路由匹配] 客户 {client_info['name']} 匹配策略类型：{client_info['client_type']}")

    # 步骤 B：执行动态路由链（支持 asyncio 异步调用）
    response = await strategy_router_branch.ainvoke(client_info)
    return response


# 6. 测试与验证
async def main():
    print("=== 测试 1：客户 CLI-001（短期投资路由） ===")
    plan_001 = await process_client_investment_plan("CLI-001")
    print(f"\n💡 [通义千问策略输出]:\n{plan_001}\n")

    print("=" * 60)

    print("=== 测试 2：客户 CLI-002（长期投资路由） ===")
    plan_002 = await process_client_investment_plan("CLI-002")
    print(f"\n💡 [通义千问策略输出]:\n{plan_002}\n")


if __name__ == "__main__":
    asyncio.run(main())
