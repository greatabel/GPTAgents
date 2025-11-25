import os
import json  # 🌟 新增：用来解析函数调用的参数 JSON
from openai import OpenAI
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", "dummy")
api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
model_name = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

# Create OpenAI client pointing to local DeepSeek-compatible server
client = OpenAI(
    api_key=api_key,
    base_url=api_base,
)

# 🌟 新增：真正执行推荐逻辑的本地 Python 函数
def recommend(topic: str, rating: str = "good"):
    """
    这是“真实”的业务函数，不是 LLM 里的 tools 定义。
    LLM 只会告诉我们：要调用 recommend，并给出 topic、rating，
    然后由我们在这里执行真正的逻辑。
    """
    # 为了简单起见，这里写死几个示例，你可以随便改
    if "time travel" in topic.lower():
        movie = "Back to the Future"
    else:
        movie = "Interstellar"

    return f"[{rating}] 推荐与『{topic}』相关的电影：{movie}"


# Example function to query ChatGPT (or DeepSeek)
def ask_chatgpt(user_message):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "recommend",
                    "description": "Provide a recommendation for any topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "The topic the user wants a recommendation for.",
                            },
                            "rating": {
                                "type": "string",
                                "description": "The rating this recommendation was given.",
                                "enum": ["good", "bad", "terrible"]
                            },
                        },
                        "required": ["topic"],
                    },
                },
            }
        ]
    )
    print('ask_chatgpt=>', response.choices[0].message.tool_calls[0].function)

    message = response.choices[0].message

    # 🌟 新增：如果模型返回了函数调用信息，就真的去调用本地的 recommend()
    if message.tool_calls:
        tool_call = message.tool_calls[0]  # 这里只取第一个调用，够你测试了
        func_name = tool_call.function.name
        args_json = tool_call.function.arguments  # 是一个 JSON 字符串
        args = json.loads(args_json)  # 解析成 Python 字典

        if func_name == "recommend":
            # 从参数里取出 topic 和 rating，然后调用我们上面定义的 recommend()
            topic = args.get("topic", "")
            rating = args.get("rating", "good")
            result = recommend(topic=topic, rating=rating)
            return result

    # 🌟 新增：如果模型没有调用任何工具，就直接返回模型的自然语言回答
    return message.content


# Example usage
user = "Can you please recommend me a time travel movie?"
response = ask_chatgpt(user)
print("1:", response)

user = "Can you please recommend me a good time travel movie?"
response = ask_chatgpt(user)
print("2:", response)
