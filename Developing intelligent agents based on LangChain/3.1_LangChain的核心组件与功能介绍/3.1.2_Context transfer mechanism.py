import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# 1. 初始化通义千问模型
llm = ChatTongyi(
    model=os.getenv("LLM_MODEL_ID", "qwen-max"),
    dashscope_api_key=os.getenv("LLM_API_KEY"),
    temperature=0.7
)

# 2. 定义 Prompt 模板，明确留出 {history} 上下文占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个金融与订单服务领域的智能助手。"),
    MessagesPlaceholder(variable_name="history"),  # 关键点：历史对话记录会被自动填充在这里传递给大模型
    ("human", "{input}")
])

# 3. 内存字典存储 Session
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 4. 构建带记忆的任务链
chain = prompt | llm

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history" # 必须与 PromptTemplate 里的 variable_name 一致
)

config = {"configurable": {"session_id": "session_user_123"}}

# --- 第一轮对话 ---
print("=== 交互 1 ===")
res1 = with_message_history.invoke({"input": "我的订单号是 ORD-20260930-888。"}, config=config)
print("🤖 Agent:", res1.content)

# 显式查看当前已积累的上下文内存
current_history = get_session_history("session_user_123")
print("\n🔍 [上下文追踪] 当前已保存的历史消息数:", len(current_history.messages))

# --- 第二轮对话 ---
print("\n=== 交互 2 ===")
res2 = with_message_history.invoke({"input": "我的订单号是多少？"}, config=config)
print("🤖 Agent:", res2.content)

# 显式打印完整的上下文传递列表
print("\n📜 [显式上下文传递验证] 发送给大模型的完整对话历史:")
for msg in current_history.messages:
    print(f"  - [{msg.type.upper()}]: {msg.content}")
