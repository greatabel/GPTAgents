import os
from openai import OpenAI
from dotenv import load_dotenv

# 1️⃣ 载入环境变量
load_dotenv()

# 2️⃣ 从 .env 文件读取自定义 API key
api_key = os.getenv("DEEPSEEK_OPENAI_API_KEY")
if not api_key:
    raise ValueError("No API key found. Please check your .env file.")
# print('api_key=', api_key)
# 3️⃣ 自定义 API Base URL （你的本地或代理服务）
api_base = "http://10.248.60.236:5000/v1"

# 4️⃣ 初始化客户端
client = OpenAI(
    api_key=api_key,
    base_url=api_base  # ✅ 指定自定义 API 地址
)

# 5️⃣ 封装一个函数与模型对话
def ask_chatgpt(user_message, user_id="5fd6b79b-9584-488a-9aaa-7825fa347703"):
    response = client.chat.completions.create(
        model="h_queue_test",  # ✅ 使用你的模型名称
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        user=user_id,
        stream=False,  # ❌ 如果你不需要流式输出，设为 False
    )
    return response.choices[0].message.content

# 6️⃣ 示例调用
if __name__ == "__main__":
    user_input = "法国首都在哪儿？"
    answer = ask_chatgpt(user_input)
    print("🤖 Deepseek:", answer)
