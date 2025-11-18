# test_with_proxy.py
"""
使用本地代理服务器测试 DeepSeek V3.2-Exp 的工具调用功能

前置条件：
1. 代理服务器正在运行: python deepseek_proxy_server.py
2. 远程 SGLang 服务可访问: http://10.248.60.236:5000
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
    # ⭐ 关键配置：使用本地代理服务器
    BASE_URL = "http://localhost:8000/v1"
    
    print("="*60)
    print("🧪 测试 DeepSeek V3.2-Exp 工具调用")
    print("="*60)
    print(f"🔗 代理服务器: {BASE_URL}")
    print(f"🌐 远程 API: http://10.248.60.236:5000")
    print(f"👤 用户: {os.getenv('USER', 'greatabel')}")
    print(f"🕐 时间: 2025-11-18 08:03:07 UTC")
    print("="*60)
    print()
    
    # 配置模型信息
    model_info = ModelInfo(
        vision=False,
        function_calling=True,  # 启用函数调用
        json_output=True,
        family="deepseek"
    )
    
    # 创建模型客户端（指向本地代理）
    model_client = OpenAIChatCompletionClient(
        model="deepseek-chat",
        api_key="dummy",  # 代理服务器会使用正确的 API key
        base_url=BASE_URL,
        model_info=model_info
    )
    
    # 定义工具
    myecho_tool = FunctionTool(
        myecho, 
        description="Echoes back the input text"
    )
    
    # 创建助手智能体
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[myecho_tool],
        system_message="You are a helpful assistant with access to tools."
    )
    
    # 设置终止条件（最多 3 条消息）
    termination = MaxMessageTermination(3)
    
    # 创建团队
    team = RoundRobinGroupChat([assistant], termination_condition=termination)
    
    # 运行测试
    print("🚀 开始测试...\n")
    
    await Console(team.run_stream(task='call myecho with text "HELLO WORLD"'))
    
    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())