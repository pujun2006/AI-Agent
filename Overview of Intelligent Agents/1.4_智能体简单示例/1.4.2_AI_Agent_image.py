import os
import dashscope
from dashscope import ImageSynthesis
from dotenv import load_dotenv

# 1. 加载 .env 环境变量
load_dotenv()

# 2. 显式设置 dashscope 的全局 API Key
dashscope.api_key = os.getenv("LLM_API_KEY")

# 检查 key 是否成功加载
if not dashscope.api_key:
    raise ValueError("未读取到 API Key，请检查 .env 文件中的 LLM_API_KEY 配置！")

# 3. 发起文生图请求
rsp = ImageSynthesis.call(
    model="wanx-v1",  # 通义万相标准文生图模型
    prompt="一幅展现编程中“递归”概念的艺术画，无限延伸的层次结构，赛博朋克风格",
    n=1,
    size="1024*1024"  # 注意：必须是星号 "*"，不能是 "x"
)

# 4. 判断并输出结果
if rsp.status_code == 200:
    print("生成成功！图片地址为：")
    print(rsp.output.results[0].url)
else:
    print(f"调用失败，错误代码: {rsp.code}, 错误信息: {rsp.message}")
