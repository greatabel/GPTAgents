import os
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

    # Return the function call (if any)
    return response.choices[0].message.tool_calls[0].function


# Example usage
user = "Can you please recommend me a time travel movie?"
response = ask_chatgpt(user)
print(response)

user = "Can you please recommend me a good time travel movie?"
response = ask_chatgpt(user)
print(response)
