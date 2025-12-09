# test_remote_deepseek.py
"""
直接测试远程 DeepSeek / SGLang OpenAI-compatible API
"""

import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai._model_info import ModelInfo
from autogen_core.tools import FunctionTool


def myecho(text: str) -> str:
    """Echoes back the input text."""
    result = f"<<ECHO: {text}>>"
    print(f"\n{'='*60}")
    print(f"🎯🎯🎯 函数被执行了！🎯🎯🎯")
    print(f"{'='*60}")
    print(f"输入: {text}")
    print(f"输出: {result}")
    print(f"{'='*60}\n")
    return result


async def main():
    # ✅ 直接连远程 SGLang / DeepSeek OpenAI API
    BASE_URL = "http://10.248.60.236:5000/v1"
    API_KEY = os.environ["DEEPSEEK_OPENAI_API_KEY"]

    print("=" * 60)
    print("🧪 测试远程 DeepSeek V3.x 工具调用")
    print("=" * 60)
    print(f"🔗 API 地址: {BASE_URL}")
    print(f"🔑 API Key: 来自环境变量 DEEPSEEK_OPENAI_API_KEY")
    print("=" * 60)
    print()

    # ✅ 模型能力声明（非常重要）
    model_info = ModelInfo(
        vision=False,
        function_calling=True,
        json_output=True,
        family="deepseek"
    )

    # ✅ OpenAI-compatible 客户端
    model_client = OpenAIChatCompletionClient(
        model="deepseek-chat",
        api_key=API_KEY,
        base_url=BASE_URL,
        model_info=model_info
    )

    # ✅ 工具
    myecho_tool = FunctionTool(
        myecho,
        description="Echoes back the input text"
    )

    # ✅ Agent
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[myecho_tool],
        system_message="You are a helpful assistant with access to tools."
    )

    # ✅ termination
    termination = MaxMessageTermination(3)

    # ✅ 单 agent team
    team = RoundRobinGroupChat(
        participants=[assistant],
        termination_condition=termination
    )

    print("🚀 开始测试...\n")

    await Console(
        team.run_stream(
            task='call myecho with text "HELLO WORLD"'
        )
    )

    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
