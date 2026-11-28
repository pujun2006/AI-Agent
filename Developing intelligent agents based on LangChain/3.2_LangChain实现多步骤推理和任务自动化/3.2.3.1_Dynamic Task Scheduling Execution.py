import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

# 1. 加载环境变量
load_dotenv()

# 2. 初始化通义千问模型
llm = ChatTongyi(
    model=os.getenv("LLM_MODEL_ID", "qwen-max"),
    dashscope_api_key=os.getenv("LLM_API_KEY"),
    temperature=0.1  # 路由分类要求高确定性，建议降低采样温度
)

output_parser = StrOutputParser()

# ------------------------------------------------------------------
# Step 1: 定义各个具体的业务子链 (Sub-chains)
# ------------------------------------------------------------------
order_chain = (
    PromptTemplate.from_template("【订单系统】请处理以下订单查询请求：{query}")
    | llm
    | output_parser
)

logistics_chain = (
    PromptTemplate.from_template("【物流系统】请处理以下物流/快递查询请求：{query}")
    | llm
    | output_parser
)

default_chain = (
    PromptTemplate.from_template("【通用客服】回答以下常规问题：{query}")
    | llm
    | output_parser
)

# ------------------------------------------------------------------
# Step 2: 定义意图分类链 (Classifier Chain)
# ------------------------------------------------------------------
router_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个意图分类助手。根据用户的提问，判断其类别。\n"
               "仅输出以下关键词之一：'ORDER'（订单查询）、'LOGISTICS'（物流查询）、'OTHER'（其他）。\n"
               "严禁输出多余文字！"),
    ("human", "{query}")
])

classifier_chain = router_prompt | llm | output_parser

# ------------------------------------------------------------------
# Step 3: 使用 RunnableBranch 构建条件动态路由
# ------------------------------------------------------------------
# 条件判别格式: (lambda x: 条件判断, 匹配时执行的链)
branch = RunnableBranch(
    (lambda x: "ORDER" in x["topic"].upper(), lambda x: order_chain.invoke({"query": x["query"]})),
    (lambda x: "LOGISTICS" in x["topic"].upper(), lambda x: logistics_chain.invoke({"query": x["query"]})),
    # 默认兜底分支
    lambda x: default_chain.invoke({"query": x["query"]})
)

# 组合分类与分支调度：输入 query -> 分类 topic -> 路由分发
full_router_chain = (
    {
        "topic": classifier_chain,
        "query": lambda x: x["query"]
    }
    | branch
)

# ------------------------------------------------------------------
# Step 4: 测试运行
# ------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "帮我看一下订单 ORD-8899 到哪里了，快递单号是多少？",
        "我想修改一下我订单 ORD-8899 的送货地址",
        "你们平台的退换货政策是怎样的？"
    ]

    for q in test_queries:
        print(f"\n❓ 用户提问: {q}")
        response = full_router_chain.invoke({"query": q})
        print(f"🤖 路由响应:\n{response}")
