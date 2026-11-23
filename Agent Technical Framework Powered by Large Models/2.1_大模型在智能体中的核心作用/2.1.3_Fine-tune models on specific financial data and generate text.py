from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


financial_corpus = """债券投资的特点是收益相对稳定、风险较低、具有固定的利息支付和到期还本的机制。
股票投资代表对公司的所有权，具备高收益与高风险并存的特征，长期来看具有抗通胀属性。
公募基金是通过公开发售基金份额募集资金，由基金管理人管理、基金托管人托管的集合投资方式。
资产配置是指根据投资者的风险承受能力和收益目标，在股票、债券、现金等不同资产类别间分配资金。
市盈率（P/E）是股票价格与每股收益的比率，常用于衡量股票估值水平的高低。
复利效应被称为世界第八大奇迹，其核心在于利息产生利息，时间越长资本累积越显著。
固定收益类理财产品主要投资于国债、金融债、企业债等低风险金融工具，适合稳健型投资者。
货币市场基金主要投资于短期有价证券，具有极高的流动性和极低违约风险，通常被视作现金替代品。
系统性风险是指由于全局性因素（如宏观经济衰退、利率变动）引起的整个金融市场的波动，无法通过分散投资消除。
非系统性风险是特定公司或行业独有的风险，可以通过构建多元化的投资组合加以规避。
通货膨胀会导致货币购买力下降，持有过多现金资产在通胀环境中会承受实际购买力缩水的损失。
投资组合的夏普比率（Sharpe Ratio）用于评估单位总风险下所获取的超额收益，数值越高表明风险调整后的绩效越好。
美联储加息通常会导致市场流动性收紧，压低风险资产价格，同时推升美元指数和美债收益率。
看涨期权给予买方在未来某一特定时间内以特定价格购买标的资产的权利，而非义务。
杠杆交易可以通过借入资金放大投资收益，但同时也会以同等比例放大亏损风险，需严格设置止损线。"""

# 写入本地文件
with open("financial_data.txt", "w", encoding="utf-8") as f:
    f.write(financial_corpus)

print("financial_data.txt 创建成功！")


# 1. 加载模型与分词器
def load_model_and_tokenizer(model_name="uer/gpt2-chinese-cluecorpussmall"):
    # 自动识别中文模型的最佳分词器类型
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()
print("模型和分词器加载成功！")


# 2. 加载与预处理数据集
def prepare_dataset(file_path, tokenizer, block_size=128):
    # 加载文本数据集
    raw_dataset = load_dataset('text', data_files={'train': file_path}, encoding='utf-8')

    def tokenize_function(examples):
        # 移除空行，避免产生空序列
        texts = [text.strip() for text in examples['text'] if len(text.strip()) > 0]
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        return tokenizer(texts, truncation=True, max_length=block_size)

    # 进行映射分词并移除原始文本列
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # 提取 train 分支并过滤掉长度为 0 的样本
    train_dataset = tokenized_dataset['train'].filter(lambda x: len(x['input_ids']) > 0)

    print(f"✅ 数据集加载成功，最终有效训练样本数：{len(train_dataset)}")
    return train_dataset


file_path = "./financial_data.txt"
dataset = prepare_dataset(file_path, tokenizer)
print("数据集加载并预处理成功！")


# 3. 训练参数配置
def get_training_arguments(output_dir="./results"):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,          # 极小数据集需要大幅增加轮数
        per_device_train_batch_size=1, # 样本极少时调小 batch_size 以增加步数
        logging_steps=5,
        report_to="none"
    )


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = get_training_arguments()
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# 4. 开始训练与保存
trainer.train()
print("模型训练完成！")

model.save_pretrained("./finetuned_gpt2")
tokenizer.save_pretrained("./finetuned_gpt2")
print("微调后的模型已保存！")


# 5. 文本生成测试
def generate_text(prompt, model, tokenizer, max_length=100):
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,

        # --- 关键生成控制参数 ---
        do_sample=True,  # 开启采样
        top_k=30,  # 限制在概率最高的前30个词中选择
        top_p=0.85,  # 核采样 (Nucleus sampling)
        temperature=0.7,  # 控制随机度（越低越确定）
        repetition_penalty=1.5,  # 重复惩罚系数（1.2 ~ 1.5），防止无限死循环
        no_repeat_ngram_size=2  # 强行禁止连续出现相同的2连词
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


prompt = "债券投资的特点是"
generated_text = generate_text(prompt, model, tokenizer)
print("生成的文本：", generated_text)