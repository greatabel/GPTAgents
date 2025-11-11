import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# 1️⃣ 加载环境变量
load_dotenv()

# 2️⃣ 从 .env 文件读取 API Key
api_key = os.getenv("DEEPSEEK_OPENAI_API_KEY")
if not api_key:
    raise ValueError("No API key found. Please check your .env file.")

# 3️⃣ DeepSeek 本地部署的 API 地址（修改为你的实际地址）
# 示例： "http://localhost:8000/v1" 或 "https://deepseek.yourdomain.com/v1"
api_base = "http://10.248.60.236:5000/v1"

# 4️⃣ 初始化客户端
client = OpenAI(
    api_key=api_key,
    base_url=api_base,  # ✅ 关键：指定自定义 API 地址
)

# 5️⃣ 定义对话函数
def ask_chatgpt(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",  # ✅ 使用 DeepSeek 模型名（例如 deepseek-chat、deepseek-coder 等）
        messages=messages,
        temperature=0.7,
    )

    # 打印完整返回 JSON
    response_model = response.model_dump()
    print(json.dumps(response_model, indent=4, ensure_ascii=False))

    # 返回模型回复内容
    return response.choices[0].message.content


# 6️⃣ 示例对话
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is an interesting fact about Paris?"}
]

if __name__ == "__main__":
    response = ask_chatgpt(messages)
    print("\n🤖 Model reply:", response)
