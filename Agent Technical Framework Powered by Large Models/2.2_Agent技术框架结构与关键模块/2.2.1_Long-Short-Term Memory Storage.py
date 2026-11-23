import sqlite3
import datetime
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch


# ==========================================
# 1. 记忆系统（Memory System）
# ==========================================
class AgentMemory:
    def __init__(self, db_path="customer_memory.db"):
        # 短期记忆：存储当前会话的上下文状态（内存字典）
        self.short_term_memory = {
            "account_id": None,
            "last_intent": None,
            "current_step": None
        }

        # 长期记忆：存储历史交互记录（SQLite 数据库）
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        """初始化长期记忆数据库"""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            query TEXT,
            agent_response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self.conn.commit()

    def update_short_term(self, key, value):
        """更新短期记忆"""
        self.short_term_memory[key] = value

    def get_short_term(self, key):
        """获取短期记忆"""
        return self.short_term_memory.get(key)

    def save_long_term(self, customer_id, query, response):
        """持久化保存到长期记忆"""
        self.cursor.execute(
            "INSERT INTO customer_history (customer_id, query, agent_response) VALUES (?, ?, ?)",
            (customer_id, query, response)
        )
        self.conn.commit()

    def get_long_term_history(self, customer_id, limit=3):
        """查询客户的长期历史记录"""
        self.cursor.execute(
            "SELECT query, agent_response, timestamp FROM customer_history WHERE customer_id = ? ORDER BY id DESC LIMIT ?",
            (customer_id, limit)
        )
        return self.cursor.fetchall()


# ==========================================
# 2. 工具库（Tools System）
# ==========================================
class FinancialTools:
    @staticmethod
    def query_credit_card_bill(account_id):
        """模拟外部 API：查询信用卡账单工具"""
        # 实际开发中此处接入银行或核心数据库 API
        mock_bills = {
            "12345678": {"amount": "¥ 3,520.50", "due_date": "2026-10-15", "status": "未还款"},
            "88888888": {"amount": "¥ 0.00", "due_date": "2026-10-20", "status": "已还清"}
        }
        return mock_bills.get(account_id, None)


# ==========================================
# 3. 智能体核心大脑（Financial Agent Core）
# ==========================================
class FinancialAgent:
    def __init__(self, model_path="./finetuned_gpt2", customer_id="12345678"):
        self.customer_id = customer_id
        self.memory = AgentMemory()

        # 默认将账号写入短期记忆
        self.memory.update_short_term("account_id", customer_id)

        # 加载大模型大脑（如果本地不存在微调权重，降级使用开源基础模型）
        print("🧠 正在初始化智能体大脑...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
        except Exception:
            print("⚠️ 未找到本地微调模型，自动回退加载开源中文预训练模型...")
            model_name = "uer/gpt2-chinese-cluecorpussmall"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _intent_recognition(self, user_query):
        """简单的意图识别路由（规划/决策层）"""
        if "账单" in user_query or "欠款" in user_query or "还款" in user_query:
            return "QUERY_BILL"
        else:
            return "GENERAL_FINANCE_QA"

    def _generate_llm_response(self, prompt, max_length=100):
        """调用微调大模型生成知识问答"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=30,
            top_p=0.85,
            temperature=0.7,
            repetition_penalty=1.3
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def handle_message(self, user_query):
        """智能体处理用户消息的主入口"""
        print(f"\n👤 用户 [{self.customer_id}]: {user_query}")

        # 1. 识别意图
        intent = self._intent_recognition(user_query)
        self.memory.update_short_term("last_intent", intent)

        response = ""

        # 2. 执行决策与行动 (Action)
        if intent == "QUERY_BILL":
            # 从短期记忆中提取当前用户的账号
            account_id = self.memory.get_short_term("account_id")

            # 调用外部工具查询数据
            bill_info = FinancialTools.query_credit_card_bill(account_id)

            if bill_info:
                response = (f"为您查询到账户 [{account_id}] 的最新信用卡账单：\n"
                            f"• 本期应还金额：{bill_info['amount']}\n"
                            f"• 到期还款日：{bill_info['due_date']}\n"
                            f"• 还款状态：{bill_info['status']}")
            else:
                response = f"抱歉，未找到账号 [{account_id}] 的账单信息。"

        elif intent == "GENERAL_FINANCE_QA":
            # 调用大模型生成金融概念回答
            response = self._generate_llm_response(user_query)

        # 3. 更新记忆系统
        self.memory.save_long_term(self.customer_id, user_query, response)

        return response


# ==========================================
# 4. 运行与交互测试
# ==========================================
if __name__ == "__main__":
    # 初始化智能体（客户ID为 12345678）
    agent = FinancialAgent(customer_id="12345678")

    # 模拟场景 1：调用工具（查询账单）
    query1 = "帮我查一下这个月的信用卡账单"
    reply1 = agent.handle_message(query1)
    print(f"🤖 Agent:\n{reply1}")

    # 模拟场景 2：调用大脑（金融知识问答）
    query2 = "债券投资的特点是"
    reply2 = agent.handle_message(query2)
    print(f"🤖 Agent:\n{reply2}")

    # 模拟场景 3：查看长期记忆数据库中的历史记录
    print("\n-------------------------------------------")
    print("📜 智能体调取长期记忆（历史咨询记录）：")
    history = agent.memory.get_long_term_history("12345678")
    for item in history:
        print(f"[{item[2]}] 问: {item[0]} --> 答: {item[1][:25]}...")