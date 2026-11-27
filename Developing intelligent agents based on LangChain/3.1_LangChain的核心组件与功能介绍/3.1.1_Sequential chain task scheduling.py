# 新架构
import os
from langchain_community.chat_models import ChatTongyi  # 替换这里
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 2. 初始化通义千问模型
llm = ChatTongyi(
    model=os.getenv("LLM_MODEL_ID"),
    dashscope_api_key=os.getenv("LLM_API_KEY"),
    temperature=0.7
)

# 3. 创建提示模板
template_1 = PromptTemplate(
    input_variables=["question"],
    template="解析问题: {question}"
)

template_2 = PromptTemplate(
    input_variables=["answer"],
    template="基于答案：{answer}，生成后续步骤"
)

# 4. 构建 LCEL 任务链
chain = (
    template_1
    | llm
    | StrOutputParser()
    | (lambda text: {"answer": text})
    | template_2
    | llm
    | StrOutputParser()
)

# 5. 执行测试
response = chain.invoke({"question": "如何实现机器学习模型的训练？"})
print(response)