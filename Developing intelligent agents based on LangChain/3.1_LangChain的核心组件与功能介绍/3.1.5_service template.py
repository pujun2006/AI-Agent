from langchain_core.prompts import PromptTemplate
# 定义客户服务模板
template = PromptTemplate(
    input_variables=["product", "issue"],
    template="请帮忙查询{product}的当前状态，并解决以下问题：{issue}"
)
# 在调用时用具体内容替换占位符
prompt = template.format(product="智能手表", issue="无法开机")
print(prompt)