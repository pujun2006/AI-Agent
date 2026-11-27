import time
import sqlite3
import asyncio
import requests
import numpy as np
from functools import wraps
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, GPT2LMHeadModel
from typing import Callable

# 全局数据库连接管理（修复1：避免频繁建立连接）
LONG_TERM_DB = "customer_memory.db"


class MemoryManager:
    """管理上下文和记忆模块"""

    def __init__(self, db_path=LONG_TERM_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customer_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT, 
                    query TEXT, 
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save_to_memory(self, customer_id: str, query: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO customer_history (customer_id, query) VALUES (?, ?)",
                (customer_id, query)
            )
            conn.commit()

    def get_last_query(self, customer_id: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT query FROM customer_history WHERE customer_id = ? ORDER BY id DESC LIMIT 1",
                (customer_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else "No history found."


memory_manager = MemoryManager()


def performance_monitor(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"⏱️ [{func.__name__}] 执行耗时: {elapsed:.2f}s")
        return result

    return wrapper


class ModelManager:
    """管理模型加载和生成"""

    def __init__(self):
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    @staticmethod
    def load_model_and_tokenizer():
        print("🧠 正在加载大模型与分词器...")
        # 换用支持中文的模型名称（如果本地已有模型路径可替换）
        model_name = "uer/gpt2-chinese-cluecorpussmall"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
        except Exception:
            # 降级备用加载原生 gpt2
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")

        # 修复2：设置 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✅ 大模型加载成功！")
        return model, tokenizer

    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        # 引导续写方向，比裸词组更易触发合理中文续写
        formatted_prompt = f"{prompt}是指" if not prompt.endswith(("？", "?", "！", "!", "：")) else prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        output = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.85,
            temperature=0.7,
            repetition_penalty=1.25,  # 惩罚重复 token
            no_repeat_ngram_size=3,  # 禁止 3-gram 重复
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


model_manager = ModelManager()


class VectorSearch:
    """语义检索模块 (RAG)"""

    def __init__(self):
        print("🔍 正在初始化向量检索模型...")
        # 修复3：中文推荐使用 text2vec 或 bge 中文语义模型
        try:
            self.model = SentenceTransformer('shibing624/text2vec-base-chinese')
        except Exception:
            self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

        self.knowledge_base = [
            "股票市场是风险投资的主要渠道，高收益高风险。",
            "债券投资收益相对稳定，适合稳健和保守型投资者。",
            "外汇市场波动较大，杠杆高，适合有经验的投资者参与。"
        ]
        self.knowledge_vectors = self.model.encode(self.knowledge_base)

    @staticmethod
    def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

    def search(self, query: str) -> str:
        query_vector = self.model.encode([query])[0]
        similarities = [self.cosine_similarity(query_vector, vec) for vec in self.knowledge_vectors]
        closest_idx = np.argmax(similarities)
        return self.knowledge_base[closest_idx]


vector_search = VectorSearch()


class APIManager:
    """外部 API 工具集成"""

    @staticmethod
    def get_weather(city: str) -> str:
        # 模拟外部 API 响应（避免未填真实 API Key 导致报错）
        return f"{city}的实时天气是：22°C 晴朗，微风"


class FinancialAgent:
    """金融智能体主控流程"""

    def _extract_city(self, query: str) -> str:
        """简单的城市提权辅助函数（修复4：防止切词导致获取全句）"""
        for c in ["北京", "上海", "广州", "深圳", "香港", "杭州"]:
            if c in query:
                return c
        return "北京"  # 默认兜底城市

    @performance_monitor
    async def handle_query(self, customer_id: str, query: str):
        # 1. 记忆写入
        memory_manager.save_to_memory(customer_id, query)

        print(f"\n👤 收到客户 [{customer_id}] 的提问: {query}")

        # 2. 规则路由与工具分发
        if "天气" in query:
            city = self._extract_city(query)
            # 使用 asyncio.to_thread 避免同步网络请求卡死事件循环
            weather_info = await asyncio.to_thread(APIManager.get_weather, city)
            print(f"🤖 [工具调用] {weather_info}")

        elif "投资" in query or "建议" in query:
            # 向量数据库匹配
            advice = await asyncio.to_thread(vector_search.search, query)
            print(f"🤖 [RAG 检索建议]: {advice}")

        else:
            # 调用底层大模型生成文本
            response = await asyncio.to_thread(model_manager.generate_text, query)
            print(f"🤖 [LLM 智能生成]: {response}")


async def main():
    agent = FinancialAgent()
    await agent.handle_query("customer_001", "请问北京的天气如何？")
    await agent.handle_query("customer_001", "给我一些债券相关的投资建议")
    await agent.handle_query("customer_001", "金融资产配置")


if __name__ == "__main__":
    asyncio.run(main())