import os
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler

# 最新标准导入路径
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

# 3. 构建两阶段 LCEL 顺序日报链
# 第一阶段：汇总数据并解析核心指标
template_extract = PromptTemplate.from_template(
    "你是一位金融分析师，请对以下原始业务数据进行关键信息提取与摘要：\n{raw_data}"
)

# 第二阶段：基于解析好的摘要生成排版美观的正式日报
template_report = PromptTemplate.from_template(
    "根据以下关键业务摘要，生成一份格式规范、排版专业且包含后续建议的金融日报：\n{summary}"
)

daily_report_chain = (
        template_extract
        | llm
        | StrOutputParser()
        | (lambda summary_text: {"summary": summary_text})  # 变量对齐转接
        | template_report
        | llm
        | StrOutputParser()
)


# 4. 定义定时触发执行的Job函数
def generate_daily_report_job():
    print("⏰ [定时任务触发] 开始拉取数据并调用通义千问生成日报...")

    # 模拟从数据库或外部 API 获取的当日业务数据
    mock_data = "本日消费贷款申请量 120 笔，审批通过 95 笔，总放款金额 450 万元，坏账率控制在 0.02% 内。"

    try:
        # 执行 LCEL 链
        report_result = daily_report_chain.invoke({"raw_data": mock_data})
        print("\n================ 📄 每日金融业务分析报告 📄 ================")
        print(report_result)
        print("========================================================\n")
    except Exception as e:
        print(f"❌ 日报生成失败: {e}")


# 5. 配置并启动 BlockingScheduler
if __name__ == "__main__":
    scheduler = BlockingScheduler()

    # 示例配置：每天早上 08:00 自动触发
    # scheduler.add_job(generate_daily_report_job, 'cron', hour=8, minute=0)

    # 测试效果：配置一个每隔 30 秒执行一次的测试任务（实际上线时可注释）
    scheduler.add_job(generate_daily_report_job, 'interval', seconds=30)

    print("🚀 APScheduler 调度器已启动，等待定时任务触发...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("🛑 调度器已安全关闭")