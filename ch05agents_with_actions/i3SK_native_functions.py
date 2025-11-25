import asyncio
import os
import json

import openai
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)

# Native function decorator
from semantic_kernel.functions import kernel_function


# ===============================
# 本地 DeepSeek (OpenAI 格式)
# ===============================
api_key = os.getenv("OPENAI_API_KEY", "dummy")
api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
model_name = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

# 兼容 openai 官方 SDK
openai.api_key = api_key

# ===============================
# 创建 OpenAI Async 客户端（指向本地 DeepSeek）
# ===============================
chat_client = openai.AsyncOpenAI(
    api_key=api_key,
    base_url=api_base,  # 非官方 OpenAI，用自己的 base_url
)

# ===============================
# Kernel
# ===============================
kernel = sk.Kernel()

# 注册 ChatCompletion 服务（Semantic Kernel 使用）
chat_service = OpenAIChatCompletion(
    service_id="local_llm",
    ai_model_id=model_name,
    async_client=chat_client,
)

kernel.add_service(chat_service)

# 默认执行参数（温度、max_tokens 等）
execution_settings = OpenAIChatPromptExecutionSettings(
    service_id="local_llm",
    ai_model_id=model_name,
    max_tokens=1500,
    temperature=0.7,
)


# ===============================
# 原生函数：从文件读取已看过电影
# ===============================
class MySeenMoviesDatabase:
    @kernel_function(
        description="Loads a list of movies the user has already seen",
        name="LoadSeenMovies",
    )
    def load_seen_movies(self) -> str:
        try:
            with open("seen_movies.txt", "r", encoding="utf-8") as f:
                return ", ".join([line.strip() for line in f.readlines()])
        except Exception as e:
            print("Error reading file:", e)
            return ""


# 把原生插件挂到 Kernel 上
seen_movies_plugin = kernel.add_plugin(
    plugin=MySeenMoviesDatabase(),
    plugin_name="SeenMoviesPlugin",
)

# 方便调用
load_seen_movies = seen_movies_plugin["LoadSeenMovies"]


async def show_seen_movies():
    result = await kernel.invoke(load_seen_movies)
    return result.value


# ===============================
# 加载 Prompt 插件（推荐电影）
#   目录结构要求：
#   plugins/
#     Recommender/
#       Recommend_Movies/
#         skprompt.txt
#         config.json
# ===============================
# 让 SK 自动从目录加载 prompt functions
recommender_plugin = kernel.add_plugin(
    parent_directory="plugins",   # 上级目录
    plugin_name="Recommender",    # 子目录名
)

# 获取具体的函数：Recommend_Movies
recommend = recommender_plugin["Recommend_Movies"]

# 绑定默认的执行参数（按服务 ID 映射）
# 注意：键名要与 service_id 一致，这里是 "local_llm"
recommend.prompt_execution_settings = {
    "local_llm": execution_settings,
}


# ===============================
# 主流程
# ===============================
async def run():
    print("=== Loading seen movies from file ===")
    seen_movies = await show_seen_movies()
    print(seen_movies)

    # 调用推荐函数，把“看过的电影列表”作为 input 变量传入
    result = await recommend(
        kernel=kernel,
        input=seen_movies,
    )

    print("\n=== Recommended Movies ===\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(run())
