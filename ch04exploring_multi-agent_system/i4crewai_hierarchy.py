# i6_game_builder_hierarchical_deepseek.py
"""
CrewAI 游戏构建器 - 层级模式 + DeepSeek API

功能：
- 使用 Process.hierarchical（层级管理模式）
- Manager LLM 协调 3 个工程师 Agent
- 使用本地 DeepSeek API
- 集成 AgentOps 监控

关键概念：
- hierarchical 模式：有一个 Manager 自动分配任务给 Agents
- Manager 需要独立的 LLM 配置
- Agents 不需要预先指定 Task（Manager 会动态分配）

前提：
1. DeepSeek 代理服务器运行在 http://localhost:8000
2. .env 文件已配置
"""

from textwrap import dedent
import agentops
from crewai import Agent, Crew, Process, Task, LLM  # ⭐ 导入 LLM
from dotenv import load_dotenv
import os

# ============================================================
# 初始化
# ============================================================

print("="*60)
print("🎮 CrewAI 游戏构建器 - 层级管理模式（DeepSeek 版）")
print("="*60)
print(f"📅 当前时间（UTC）: 2025-11-22 00:34:26")
print(f"👤 当前用户: greatabel")
print("="*60 + "\n")

load_dotenv()  # 加载 .env 文件
agentops.init()  # 初始化 AgentOps

print("✅ AgentOps 已启动\n")

# ⭐⭐⭐ 配置本地 DeepSeek LLM ⭐⭐⭐
# 这个 LLM 配置会被所有 Agents 使用
deepseek_llm = LLM(
    model="deepseek-chat",
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    temperature=0.7,
)

# ⭐⭐⭐ Manager LLM 配置（hierarchical 模式必需）⭐⭐⭐
# Manager 负责协调和分配任务，通常需要更强的推理能力
# 这里也使用 DeepSeek（如果您有更好的模型，可以单独配置）
manager_llm = LLM(
    model="deepseek-chat",
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    temperature=0.3,  # ⭐ Manager 用较低温度（更理性、更稳定）
)

print("✅ LLM 配置:")
print(f"   Agent LLM: deepseek-chat (温度: 0.7)")
print(f"   Manager LLM: deepseek-chat (温度: 0.3)")
print(f"   地址: http://localhost:8000/v1\n")

# ============================================================
# 用户输入
# ============================================================

print("## 欢迎来到游戏创作工坊（层级管理模式）")
print("-"*60)
print("💡 提示：在层级模式中，Manager 会自动协调 3 个工程师的工作")
print("-"*60)
game = input("🎮 您想创建什么游戏？游戏机制是什么？\n请用中文或英文描述：\n> ")

if not game.strip():
    game = "Create a simple Snake game using Pygame with score tracking"
    print(f"\n💡 使用默认示例: {game}\n")

print("\n" + "="*60)
print("🚀 开始构建游戏...")
print("="*60 + "\n")

# ============================================================
# 创建 Agents（⭐ 在 hierarchical 模式中，不需要在 Task 中指定 agent）
# ============================================================

# Agent 1: 资深软件工程师
senior_engineer_agent = Agent(
    role="Senior Software Engineer",
    goal="Create high-quality, production-ready Python software",
    backstory=dedent(
        """
        You are a Senior Software Engineer at a leading tech think tank.
        Your expertise in Python programming is exceptional, and you produce
        clean, well-documented, and efficient code.
        You follow best practices and write code that is maintainable.
        """
    ),
    allow_delegation=False,  # 在 hierarchical 模式中，通常设为 False
    verbose=False,           # 减少日志
    memory=False,            # 避免 Embeddings 404
    llm=deepseek_llm,       # ⭐ 使用 DeepSeek
)

# Agent 2: 质量控制工程师
qa_engineer_agent = Agent(
    role="Software Quality Control Engineer",
    goal="Create perfect code by analyzing for errors and vulnerabilities",
    backstory=dedent(
        """
        You are a software engineer specializing in code quality and testing.
        You have an eagle eye for detail and excel at finding hidden bugs.
        You meticulously check for:
        - Missing imports and dependencies
        - Variable declarations and scope issues
        - Mismatched brackets and syntax errors
        - Security vulnerabilities
        - Logic errors and edge cases
        - Performance bottlenecks
        """
    ),
    allow_delegation=False,
    verbose=False,
    memory=False,
    llm=deepseek_llm,
)

# Agent 3: 首席质量官
chief_qa_engineer_agent = Agent(
    role="Chief Software Quality Control Engineer",
    goal="Ensure the code meets all requirements and quality standards",
    backstory=dedent(
        """
        You are the Chief Software Quality Control Engineer at a leading
        tech think tank. You are the final authority on code quality.
        You verify that:
        - The code fulfills all specified requirements
        - All bugs and issues have been resolved
        - The code is production-ready and maintainable
        - Best practices are followed throughout
        You provide the final, polished version of the code.
        """
    ),
    allow_delegation=False,  # ⭐ 在 hierarchical 中，Manager 负责协调
    verbose=False,
    memory=False,
    llm=deepseek_llm,
)

print("✅ 已创建 3 个 Agents:")
print("   1. Senior Engineer (编写代码)")
print("   2. QA Engineer (检查错误)")
print("   3. Chief QA Engineer (最终审核)")
print("\n💡 Manager 会自动协调他们的工作\n")

# ============================================================
# 创建 Tasks（⭐ hierarchical 模式：不指定 agent！）
# ============================================================

# Task 1: 编写代码
code_task = Task(
    description=dedent(f"""
        Create a complete Python game based on these instructions:
        
        Instructions:
        ------------
        {game}
        
        Requirements:
        - Write complete, runnable Python code
        - Include all necessary imports
        - Add clear comments explaining the logic
        - Follow Python best practices (PEP 8)
        - Handle edge cases and potential errors
        - Make the code beginner-friendly if applicable
        
        Your code should be production-ready.
    """),
    expected_output="The full Python code, only the Python code and nothing else. No explanations, no markdown formatting.",
    # ⭐ 注意：在 hierarchical 模式中，不指定 agent！
    # Manager 会自动选择合适的 Agent
)

# Task 2: 质量检查
qa_task = Task(
    description=dedent(f"""
        You are helping create a game using Python. Instructions:
        
        Instructions:
        ------------
        {game}
        
        Review the code produced by the Senior Engineer.
        Check thoroughly for:
        1. Logic errors
        2. Syntax errors
        3. Missing imports
        4. Variable declaration issues
        5. Mismatched brackets or parentheses
        6. Security vulnerabilities
        7. Performance issues
        8. Code style violations
        
        Be specific and cite line numbers or code snippets where possible.
    """),
    expected_output="A detailed list of issues found in the code. If no issues, state 'No issues found. Code is clean.'",
    # ⭐ 不指定 agent
)

# Task 3: 最终审核和修正
evaluate_task = Task(
    description=dedent(f"""
        You are the final authority on this Python game project. Instructions:
        
        Instructions:
        ------------
        {game}
        
        Your responsibilities:
        1. Review the original code from the Senior Engineer
        2. Review the QA report from the QA Engineer
        3. Fix all identified issues
        4. Ensure the code meets ALL requirements
        5. Polish the code for production readiness
        
        Produce the final, corrected, complete version.
    """),
    expected_output="The final, corrected, complete Python code. Only the code, nothing else. No explanations, no markdown.",
    output_file="game_code_final.py",  # ⭐ 保存最终版本
    # ⭐ 不指定 agent
)

print("✅ 已创建 3 个 Tasks:")
print("   1. Code Task (编写游戏代码)")
print("   2. QA Task (检查代码错误)")
print("   3. Evaluate Task (最终审核修正)")
print("\n💡 Manager 会决定由谁执行每个任务\n")

# ============================================================
# 创建 Crew（⭐ hierarchical 模式）
# ============================================================

crew = Crew(
    agents=[senior_engineer_agent, qa_engineer_agent, chief_qa_engineer_agent],
    tasks=[code_task, qa_task, evaluate_task],
    verbose=1,  # 1 或 2（1: 正常，2: 详细）
    process=Process.hierarchical,  # ⭐ 层级模式
    manager_llm=manager_llm,        # ⭐ Manager 使用的 LLM（必需）
    memory=False,                   # 禁用团队记忆
    # ⭐ 注意：hierarchical 模式中，不能设置 cache
)

print("✅ 已创建 Crew（层级管理模式）")
print("   - Manager 会协调所有工作")
print("   - Tasks 会被动态分配给最合适的 Agent\n")

# ============================================================
# 执行任务
# ============================================================

print("="*60)
print("⚙️  团队开始工作（Manager 正在协调）...")
print("="*60)
print("📝 提示：")
print("   - 层级模式可能需要更长时间（3-8 分钟）")
print("   - Manager 会思考如何最优分配任务")
print("   - 请耐心等待...\n")

try:
    result = crew.kickoff()
    
    print("\n" + "="*60)
    print("🎉 游戏代码已生成！")
    print("="*60)
    print("\n📄 最终代码:\n")
    print("-"*60)
    print(result)
    print("-"*60)
    
    # 检查输出文件
    if os.path.exists("game_code_final.py"):
        print("\n✅ 代码已保存到: game_code_final.py")
        print("💡 运行游戏: python game_code_final.py\n")
    
    print("="*60)
    print("✅ 任务完成！")
    print("="*60)
    print("\n💡 提示:")
    print("   - 查看 AgentOps 控制台获取详细执行追踪")
    print("   - 在层级模式中，可以看到 Manager 的决策过程")
    print("   - 如果代码有问题，尝试提供更详细的需求\n")

except Exception as e:
    print("\n" + "="*60)
    print("❌ 执行出错")
    print("="*60)
    print(f"错误信息: {e}")
    print("\n💡 可能的原因:")
    print("   1. DeepSeek 代理服务器未启动")
    print("   2. Manager LLM 配置问题")
    print("   3. 层级模式需要更强的模型推理能力")
    print("\n解决方法:")
    print("   - 确保 deepseek_proxy_server.py 正在运行")
    print("   - 检查 .env 文件配置")
    print("   - 查看代理服务器日志")
    print("   - 考虑使用 Process.sequential 模式（更简单）\n")
    
    import traceback
    print("\n详细错误:")
    traceback.print_exc()