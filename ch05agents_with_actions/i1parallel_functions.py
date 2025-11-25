import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ===== 本地 LLM 配置 =====
api_key = os.getenv("OPENAI_API_KEY", "dummy")
api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
model_name = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

client = OpenAI(
    api_key=api_key,
    base_url=api_base,  # 注意这里参数名是 base_url
)


def recommend(topic, rating="good"):
    """Give a recommendation for any topic"""
    if "time travel" in topic.lower():
        return json.dumps({
            "topic": "time travel",
            "recommendation": "Back to the Future",
            "rating": rating
        })
    elif "recipe" in topic.lower():
        return json.dumps({
            "topic": "recipe",
            "recommendation": "The best thing you ever ate.",
            "rating": rating
        })
    elif "gift" in topic.lower():
        return json.dumps({
            "topic": "gift",
            "recommendation": "A glorius new...",
            "rating": rating
        })
    else:
        return json.dumps({
            "topic": topic,
            "recommendation": "unknown"
        })


def run_conversation():
    # Step 1: send the conversation and available functions to the model
    user = """Can you please make recommendations for the following:
    1. Time travel movies
    2. Recipes
    3. Gifts"""
    messages = [{"role": "user", "content": user}]
    tools = [
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
                            "description": "The topic, a user wants a recommnedation for.",
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

    response = client.chat.completions.create(
        model=model_name,          # 使用环境变量里的模型名
        messages=messages,
        tools=tools,
        tool_choice="auto",        # auto is default, but we'll be explicit
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        available_functions = {
            "recommend": recommend,
        }
        messages.append(response_message)  # extend conversation with assistant's reply

        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                topic=function_args.get("topic"),
                rating=function_args.get("rating"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        print('messages:==>\n', messages)
        second_response = client.chat.completions.create(
            model=model_name,      # 这里也换成本地模型名
            messages=messages,
        )
        return second_response.choices[0].message.content


if __name__ == "__main__":
    print(run_conversation())
