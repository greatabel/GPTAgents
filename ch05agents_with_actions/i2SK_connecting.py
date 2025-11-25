import asyncio
import os
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# -------- 本地 DeepSeek 配置 --------
api_key = os.getenv("OPENAI_API_KEY", "dummy")
api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
model_name = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

# 兼容 OpenAI 官方 SDK 的环境变量名字
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = api_base          # 新版 openai 用这个
os.environ["OPENAI_API_BASE"] = api_base          # 兼容一些旧代码 / 你的习惯
os.environ["OPENAI_CHAT_MODEL_ID"] = model_name   # SK 默认会用到

kernel = sk.Kernel()

service_id = "local_deepseek"

# 注意：这里不要传 endpoint / base_url，否则就会 TypeError
kernel.add_service(
    OpenAIChatCompletion(
        service_id=service_id,
        ai_model_id=model_name,   # 例如 deepseek-chat
        api_key=api_key,
        # org_id 可以不填
    )
)

# -------- 运行一个 Prompt 测试 --------
async def run_prompt():
    result = await kernel.invoke_prompt(
        prompt="recommend a movie about time travel",
        service_id=service_id,  # 指定刚才注册的服务
    )
    print(result)

asyncio.run(run_prompt())
