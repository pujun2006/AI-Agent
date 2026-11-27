import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 加载环境变量
load_dotenv()

# 2. 初始化通义千问模型
llm = ChatTongyi(
    model=os.getenv("LLM_MODEL_ID", "qwen-max"),
    dashscope_api_key=os.getenv("LLM_API_KEY"),
    temperature=0.7
)

# 3. 构建分支子链 (LCEL)
output_parser = StrOutputParser()

loan_chain = (
    PromptTemplate.from_template("用户询问贷款问题：{user_input}。请详细解释贷款流程。")
    | llm
    | output_parser
)

credit_card_chain = (
    PromptTemplate.from_template("用户询问信用卡问题：{user_input}。请详细解释信用卡的申请条件。")
    | llm
    | output_parser
)

# 4. 路由函数逻辑
def route_task(task_type, user_input):
    if task_type == "贷款":
        return loan_chain.invoke({"user_input": user_input})
    elif task_type == "信用卡":
        return credit_card_chain.invoke({"user_input": user_input})
    else:
        return "无效的任务类型，请输入'贷款'或'信用卡'。"

# 5. 测试运行
task_type = "贷款"
user_input = "如何申请个人贷款？"
response = route_task(task_type, user_input)

print(response)
