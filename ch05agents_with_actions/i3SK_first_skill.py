import asyncio
import os
import json

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelFunctionMetadata
from semantic_kernel.functions.function_result import FunctionResult



# ===============================
# 本地 OpenAI 兼容模型设置 

# ===============================
api_key = os.getenv("OPENAI_API_KEY", "dummy")
api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
model_name = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = api_base
os.environ["OPENAI_API_BASE"] = api_base
os.environ["OPENAI_CHAT_MODEL_ID"] = model_name

# ===============================
# 初始化 Kernel & 注册服务
# ===============================
kernel = sk.Kernel()

service_id = "local_llm"

kernel.add_service(
    OpenAIChatCompletion(
        service_id=service_id,
        ai_model_id=model_name,
        api_key=api_key,
    )
)

# ===============================
# 执行设置
# ===============================
execution_settings = OpenAIChatPromptExecutionSettings(
    service_id=service_id,
    ai_model_id=model_name,
    max_tokens=2000,
    temperature=0.7,
)


# ===============================
# 手动加载插件 (config.json + skprompt.txt)
# ===============================
def load_prompt_function(plugin_folder, function_name):
    """从目录手动加载语义函数 —— 适配 SK 1.36.0"""

    function_dir = os.path.join(plugin_folder, function_name)

    # 读取 config.json
    with open(os.path.join(function_dir, "config.json"), "r") as f:
        config = json.load(f)

    # 读取 skprompt.txt
    with open(os.path.join(function_dir, "skprompt.txt"), "r") as f:
        prompt_template = f.read()

    # 构造元数据
    metadata = KernelFunctionMetadata(
        name=function_name,
        plugin_name="Recommender",
        description=config.get("description", ""),
        parameters=config["input"]["parameters"],
        is_prompt=True,
    )

    # 注册函数
    kernel.add_function(
        plugin_name="Recommender",
        function_name=function_name,
        prompt=prompt_template,
        prompt_execution_settings=execution_settings,
        metadata=metadata
    )

    return kernel.get_function("Recommender", function_name)


# 加载 Recommend_Movies 插件函数
recommend = load_prompt_function("plugins/Recommender", "Recommend_Movies")

# ===============================
# 测试调用
# ===============================
seen_movie_list = [
    "Back to the Future",
    "The Terminator",
    "12 Monkeys",
    "Looper",
    "Groundhog Day",
    "Primer",
    "Donnie Darko",
    "Interstellar",
    "Time Bandits",
    "Doctor Strange",
]


async def run():
    result = await kernel.invoke(
    recommend,
    input=", ".join(seen_movie_list),
    settings=execution_settings,
    )
    print("\n=== Result ===\n")
    print(result)


asyncio.run(run())
