# i6_game_builder_with_deepseek.py
"""
CrewAI 游戏构建器 - 使用本地 DeepSeek API

功能：
- 三个 Agent 协作开发 Python 游戏
- Senior Engineer: 编写代码
- QA Engineer: 检查错误
- Chief QA Engineer: 最终审核
- 使用本地 DeepSeek API
- 集成 AgentOps 监控

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
print("🎮 CrewAI 游戏构建器（DeepSeek 版）")
print("="*60)
print(f"📅 当前时间（UTC）: 2025-11-22 00:22:31")
print(f"👤 当前用户: greatabel")
print("="*60 + "\n")

load_dotenv()  # 加载 .env 文件
agentops.init()  # 初始化 AgentOps

print("✅ AgentOps 已启动\n")

# ⭐⭐⭐ 配置本地 DeepSeek LLM ⭐⭐⭐
deepseek_llm = LLM(
    model="deepseek-chat",
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    temperature=0.7,  # 适中的创意度
)

print("✅ LLM 配置:")
print(f"   模型: deepseek-chat")
print(f"   地址: http://localhost:8000/v1")
print(f"   温度: 0.7\n")

# ============================================================
# 用户输入
# ============================================================

print("## 欢迎来到游戏创作工坊")
print("-"*60)
game = input("🎮 您想创建什么游戏？游戏机制是什么？\n请用中文或英文描述：\n> ")

if not game.strip():
    game = "Create a simple Snake game using Pygame"
    print(f"\n💡 使用默认示例: {game}\n")

print("\n" + "="*60)
print("🚀 开始构建游戏...")
print("="*60 + "\n")

# ============================================================
# 创建 Agents（三个工程师）
# ============================================================

# Agent 1: 资深软件工程师（编写代码）
senior_engineer_agent = Agent(
    role="Senior Software Engineer",
    goal="Create high-quality Python software as needed",
    backstory=dedent(
        """
        You are a Senior Software Engineer at a leading tech think tank.
        Your expertise in programming in Python is exceptional, and you do 
        your best to produce perfect, clean, and well-documented code.
        You follow best practices and write code that is easy to understand.
        """
    ),
    allow_delegation=False,  # 不能委派（专注编码）
    verbose=False,           # ⭐ 改为 False（减少日志）
    memory=False,            # ⭐ 禁用记忆（避免 404）
    llm=deepseek_llm,       # ⭐ 使用本地 DeepSeek
)

# Agent 2: 质量控制工程师（检查错误）
qa_engineer_agent = Agent(
    role="Software Quality Control Engineer",
    goal="Create perfect code by analyzing the code for errors",
    backstory=dedent(
        """
        You are a software engineer that specializes in checking code
        for errors. You have an eagle eye for detail and a knack for finding
        hidden bugs.
        You check for:
        - Missing imports
        - Variable declarations
        - Mismatched brackets and syntax errors
        - Security vulnerabilities
        - Logic errors
        - Performance issues
        """
    ),
    allow_delegation=False,  # 不能委派（专注检查）
    verbose=False,
    memory=False,
    llm=deepseek_llm,
)

# Agent 3: 首席质量官（最终审核）
chief_qa_engineer_agent = Agent(
    role="Chief Software Quality Control Engineer",
    goal="Ensure that the code does the job it is supposed to do",
    backstory=dedent(
        """
        You are the Chief Software Quality Control Engineer at a leading
        tech think tank. You are responsible for ensuring that the code
        meets all requirements and is of the highest quality.
        You review the code for:
        - Correctness: Does it fulfill the requirements?
        - Completeness: Is anything missing?
        - Quality: Is it production-ready?
        You can delegate tasks to find issues or clarify requirements.
        """
    ),
    allow_delegation=True,   # 可以委派（协调角色）
    verbose=False,
    memory=False,
    llm=deepseek_llm,
)

print("✅ 已创建 3 个 Agents:")
print("   1. Senior Engineer (编写代码)")
print("   2. QA Engineer (检查错误)")
print("   3. Chief QA Engineer (最终审核)\n")

# ============================================================
# 创建 Tasks（三个任务）
# ============================================================

# Task 1: 编写代码
code_task = Task(
    description=dedent(f"""
        You will create a game using Python based on these instructions:
        
        Instructions:
        ------------
        {game}
        
        Requirements:
        - Write complete, runnable Python code
        - Include all necessary imports
        - Add clear comments
        - Follow Python best practices
        - Make the code beginner-friendly
        
        Your code should be production-ready and well-structured.
    """),
    expected_output="Your final answer must be the full Python code, only the Python code and nothing else.",
    agent=senior_engineer_agent,
)

# Task 2: 质量检查
qa_task = Task(
    description=dedent(f"""
        You are helping create a game using Python. Here are the instructions:
        
        Instructions:
        ------------
        {game}
        
        Using the code provided by the Senior Engineer, check for:
        1. Logic errors
        2. Syntax errors
        3. Missing imports
        4. Variable declarations
        5. Mismatched brackets
        6. Security vulnerabilities
        7. Performance issues
        8. Code style issues
        
        Be thorough and specific in your findings.
    """),
    expected_output="Output a detailed list of issues found in the code, with line numbers if possible. If no issues, say 'No issues found.'",
    agent=qa_engineer_agent,
)

# Task 3: 最终审核和修正
evaluate_task = Task(
    description=dedent(f"""
        You are helping create a game using Python. Here are the instructions:
        
        Instructions:
        ------------
        {game}
        
        Review the code and QA feedback to ensure:
        1. The code fulfills all requirements
        2. All issues found by QA are fixed
        3. The code is complete and production-ready
        4. The code is well-documented
        
        If there are issues, fix them. If the code is good, approve it.
    """),
    expected_output="Your final answer must be the corrected and complete Python code, only the Python code and nothing else.",
    agent=chief_qa_engineer_agent,
    output_file="game_code.py",  # ⭐ 保存到文件
)

print("✅ 已创建 3 个 Tasks:")
print("   1. Code Task (编写代码)")
print("   2. QA Task (检查错误)")
print("   3. Evaluate Task (最终审核)\n")

# ============================================================
# 创建 Crew（团队）
# ============================================================

crew = Crew(
    agents=[senior_engineer_agent, qa_engineer_agent, chief_qa_engineer_agent],
    tasks=[code_task, qa_task, evaluate_task],
    verbose=1,               # ⭐ 改为 1（减少输出）
    process=Process.sequential,
    memory=False,            # ⭐ 禁用团队记忆
    cache=False,             # ⭐ 禁用缓存
)

print("✅ 已创建 Crew（顺序执行）\n")

# ============================================================
# 执行任务
# ============================================================

print("="*60)
print("⚙️  团队开始工作...")
print("="*60)
print("📝 提示：这可能需要 2-5 分钟，请耐心等待...\n")

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
    if os.path.exists("game_code.py"):
        print("\n✅ 代码已保存到: game_code.py")
        print("💡 运行游戏: python game_code.py\n")
    
    print("="*60)
    print("✅ 任务完成！")
    print("="*60)
    print("\n💡 提示:")
    print("   - 查看 AgentOps 控制台获取详细执行追踪")
    print("   - 如果代码有问题，可以重新运行并提供更详细的需求\n")

except Exception as e:
    print("\n" + "="*60)
    print("❌ 执行出错")
    print("="*60)
    print(f"错误信息: {e}")
    print("\n💡 可能的原因:")
    print("   1. DeepSeek 代理服务器未启动")
    print("   2. 网络连接问题")
    print("   3. API Key 配置错误")
    print("\n解决方法:")
    print("   - 确保 deepseek_proxy_server.py 正在运行")
    print("   - 检查 .env 文件配置")
    print("   - 查看代理服务器日志\n")