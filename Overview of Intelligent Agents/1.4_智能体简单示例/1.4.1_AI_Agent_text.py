import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量
load_dotenv()

client = OpenAI(api_key=os.getenv("LLM_API_KEY"),base_url=os.getenv("LLM_BASE_URL"))
completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {
            "role": "user",
            "content": "请用中文写一首关于编程中“递归”的俳句或三行小诗。"
        }
    ]
)

# 记得加上 .content
print(completion.choices[0].message.content)
