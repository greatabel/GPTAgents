import asyncio
import os

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.functions import KernelFunctionFromPrompt, KernelArguments

# -------- 本地 DeepSeek 配置 --------
api_key = os.getenv("OPENAI_API_KEY", "dummy")
api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
model_name = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

# 兼容 OpenAI 官方 SDK 的环境变量名字
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = api_base
os.environ["OPENAI_API_BASE"] = api_base
os.environ["OPENAI_CHAT_MODEL_ID"] = model_name

kernel = sk.Kernel()

service_id = "local_deepseek"

kernel.add_service(
    OpenAIChatCompletion(
        service_id=service_id,
        ai_model_id=model_name,
        api_key=api_key,
    )
)

# set up the execution settings
execution_settings = sk_oai.OpenAIChatPromptExecutionSettings(
    service_id=service_id,
    ai_model_id=model_name,
    max_tokens=2000,
    temperature=0.7,
)

prompt = """
system:

You have vast knowledge of everything and can recommend anything provided you are given the following criteria, the subject, genre, format and any other custom information.

user:
Please recommend a {{$format}} with the subject {{$subject}} and {{$genre}}. 
Include the following custom information: {{$custom}}
"""

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="tldr",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(
            name="format", description="The format to recommend", is_required=True
        ),
        InputVariable(
            name="subject", description="The subject to recommend", is_required=True
        ),
        InputVariable(
            name="genre", description="The genre to recommend", is_required=True
        ),
        InputVariable(
            name="custom",
            description="Any custom information to enhance the recommendation",
            is_required=True,
        ),
    ],
    execution_settings=execution_settings,
)

recommend_function = KernelFunctionFromPrompt(
    prompt_template_config=prompt_template_config,
    function_name="Recommend_Movies",
    plugin_name="Recommendation",
)


async def run_recommendation(
    subject="time travel", format="movie", genre="medieval", custom="must be a comedy"
):
    recommendation = await kernel.invoke(
        recommend_function,
        KernelArguments(subject=subject, format=format, genre=genre, custom=custom),
    )
    print(recommendation)


# Use asyncio.run to execute the async function
asyncio.run(run_recommendation())