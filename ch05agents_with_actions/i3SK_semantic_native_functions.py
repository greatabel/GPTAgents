import asyncio
import os

import openai
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.functions import kernel_function
from semantic_kernel.prompt_template import PromptTemplateConfig

# ===============================
# 本地 DeepSeek (OpenAI 格式)
# ===============================
api_key = os.getenv("OPENAI_API_KEY", "dummy")
api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
model_name = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

# 兼容 openai 官方 SDK
openai.api_key = api_key

# 创建 Async 客户端（指向本地 DeepSeek）
chat_client = openai.AsyncOpenAI(
    api_key=api_key,
    base_url=api_base,  # 本地/代理地址
)

# ===============================
# 创建 Kernel & 注册 Chat 服务
# ===============================
kernel = sk.Kernel()

chat_service = OpenAIChatCompletion(
    service_id="local_llm",
    ai_model_id=model_name,
    async_client=chat_client,  # ⭐ 用自己的 async_client，而不是 env
)

kernel.add_service(chat_service)

# 默认执行参数（温度、max_tokens 等）
execution_settings = OpenAIChatPromptExecutionSettings(
    service_id="local_llm",
    ai_model_id=model_name,
    max_tokens=2000,
    temperature=0.7,
)


# ===============================
# 原生函数：读取 seen_movies.txt
# ===============================
class MySeenMoviesDatabase:
    """
    Description: Loads a list of movies the user has already seen.
    """

    @kernel_function(
        description="Loads a list of movies that the user has already seen",
        name="LoadSeenMovies",
    )
    def load_seen_movies(self) -> str:
        try:
            with open("seen_movies.txt", "r", encoding="utf-8") as file:
                lines = [line.strip() for line in file.readlines()]
            return ", ".join(lines)
        except Exception as e:
            print(f"Error reading file: {e}")
            # SK 1.x 里返回空字符串会比 None 更安全
            return ""


# ✅ SK 1.x 新写法：add_plugin，而不是 import_plugin_from_object
seen_movies_plugin = kernel.add_plugin(
    plugin=MySeenMoviesDatabase(),
    plugin_name="SeenMoviesPlugin",
)

# 如果你想单独调用这个 native function，也可以这么拿：
load_seen_movies = seen_movies_plugin["LoadSeenMovies"]


# ===============================
# 内联 Prompt（不走 plugins 目录）
# ===============================
sk_prompt = """
You are a wise movie recommender and you have been asked to recommend a movie to a user.
You have a list of movies that the user has watched before.
You want to recommend a movie that the user has not watched before.
Movie List: {{SeenMoviesPlugin.LoadSeenMovies}}.
"""

# ⭐ SK 1.3.6 推荐用 PromptTemplateConfig，然后用 kernel.add_function
prompt_template_config = PromptTemplateConfig(
    template=sk_prompt,
    name="RecommendMovies",
    template_format="semantic-kernel",
)

# 创建一个 Prompt Function（类似你之前的 create_function_from_prompt）
recommend_function = kernel.add_function(
    function_name="Recommend_Movies",
    plugin_name="Recommendation",
    prompt_template_config=prompt_template_config,
)

# 绑定执行参数；键要和 service_id 对应，这里是 local_llm
recommend_function.prompt_execution_settings = {
    "local_llm": execution_settings,
}



# ===============================
# 主执行：调用推荐函数
# ===============================
async def run_recommendation():
    # 这里不需要 KernelArguments
    # prompt 里已经会自动调用 SeenMoviesPlugin.LoadSeenMovies
    recommendation = await kernel.invoke(recommend_function)

    print("\n=== Recommended Movies ===\n")
    print(recommendation)


if __name__ == "__main__":
    asyncio.run(run_recommendation())

