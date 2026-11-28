import os
from dotenv import load_dotenv

# 最新标准核心模块导入
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. 加载环境变量 (.env)
load_dotenv()

# 2. 初始化通义千问模型
llm = ChatTongyi(
    model=os.getenv("LLM_MODEL_ID", "qwen-max"),
    dashscope_api_key=os.getenv("LLM_API_KEY"),
    temperature=0.7
)

# 3. 创建带有历史上下文占位符的对话 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的客服助手，负责解答用户的订单与业务查询。"),
    MessagesPlaceholder(variable_name="history"),  # 对话历史占位符
    ("human", "{input}")
])

# 4. 创建内存字典与 Session 历史获取函数
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 5. 构建带记忆功能的 LCEL 任务链
base_chain = prompt | llm | StrOutputParser()

conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 6. 用户输入触发执行
if __name__ == "__main__":
    # 配置会话 ID (用于区分不同用户的上下文)
    config = {"configurable": {"session_id": "user_session_001"}}

    # 触发第一轮输入
    user_input_1 = "查询订单状态"
    print(f"👤 User: {user_input_1}")

    response_1 = conversation_chain.invoke({"input": user_input_1}, config=config)
    print(f"🤖 Agent:\n{response_1}\n")

    # 触发第二轮输入（验证上下文传递）
    user_input_2 = "我的订单号是 ORD-20260930-888，帮我看看进度"
    print(f"👤 User: {user_input_2}")

    response_2 = conversation_chain.invoke({"input": user_input_2}, config=config)
    print(f"🤖 Agent:\n{response_2}")
