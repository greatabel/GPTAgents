# i5_crewai_with_agentops.py
"""
CrewAI + AgentOps 集成 - 使用本地 DeepSeek API

功能：
- 使用 AgentOps 监控和追踪
- 调用本地 DeepSeek API
- 生成 AI 工程师笑话

前提：
1. DeepSeek 代理服务器运行在 http://localhost:8000
2. .env 文件包含：
   - AGENTOPS_API_KEY=your-agentops-key
   - OPENAI_API_KEY=dummy
   - OPENAI_API_BASE=http://localhost:8000/v1
"""

import agentops
from crewai import Agent, Crew, Process, Task, LLM  # ⭐ 导入 LLM
from dotenv import load_dotenv
import os

print("="*60)
print("🤖 CrewAI + AgentOps 笑话生成器")
print("="*60)
print(f"📅 当前时间（UTC）: {os.getenv('CURRENT_DATETIME', '2025-11-21 10:52:42')}")
print(f"👤 当前用户: {os.getenv('CURRENT_USER', 'greatabel')}")
print("="*60 + "\n")

# ============================================================
# 初始化
# ============================================================

load_dotenv()  # 加载 .env 文件

'''
https://app.agentops.ai/overview 
可以去这里看AgentOps（用于监控和追踪）
'''

# 初始化 AgentOps（用于监控和追踪）
# 会自动从 .env 读取 AGENTOPS_API_KEY
print("🔧 初始化 AgentOps...")
agentops.init()
print("✅ AgentOps 已启动\n")

# ⭐⭐⭐ 配置本地 DeepSeek LLM ⭐⭐⭐
# 这是关键！告诉 CrewAI 使用本地 API 而不是 OpenAI
deepseek_llm = LLM(
    model="deepseek-chat",
    base_url="http://localhost:8000/v1",  # 本地代理地址
    api_key=os.getenv("OPENAI_API_KEY", "dummy"),  # 从 .env 读取
    temperature=0.7,
)

print("✅ LLM 配置:")
print(f"   模型: deepseek-chat")
print(f"   地址: http://localhost:8000/v1")
print(f"   温度: 0.7\n")

# ============================================================
# 创建 Agents（⭐ 添加 llm 参数）
# ============================================================

joke_researcher = Agent(
    role="Senior Joke Researcher",
    goal="Research what makes things funny about the following {topic}",
    verbose=True,
    memory=False,  # ⭐ 改为 False（避免 Embeddings 404 错误）
    backstory=(
        "Driven by slapstick humor, you are a seasoned joke researcher "
        "who knows what makes people laugh. You have a knack for finding "
        "the funny in everyday situations and can turn a dull moment into "
        "a laugh riot."
    ),
    allow_delegation=True,
    llm=deepseek_llm,  # ⭐ 添加这行！
)

joke_writer = Agent(
    role="Joke Writer",
    goal="Write a humorous and funny joke on the following {topic}",
    verbose=True,
    memory=False,  # ⭐ 改为 False
    backstory=(
        "You are a joke writer with a flair for humor. You can turn a "
        "simple idea into a laugh riot. You have a way with words and "
        "can make people laugh with just a few lines."
    ),
    allow_delegation=False,
    llm=deepseek_llm,  # ⭐ 添加这行！
)

print("✅ 已创建 2 个 Agents（使用本地 DeepSeek）\n")

# ============================================================
# 创建 Tasks
# ============================================================

research_task = Task(
    description=(
        "Identify what makes the following topic: {topic} so funny. "
        "Be sure to include the key elements that make it humorous. "
        "Also, provide an analysis of the current social trends, "
        "and how it impacts the perception of humor."
    ),
    expected_output="A comprehensive 3 paragraphs long report on the latest jokes.",
    agent=joke_researcher,
)

write_task = Task(
    description=(
        "Compose an insightful, humorous and socially aware joke on {topic}. "
        "Be sure to include the key elements that make it funny and "
        "relevant to the current social trends."
    ),
    expected_output="A concise and short one line joke on {topic}.",
    agent=joke_writer,
    async_execution=False,
    output_file="the_best_joke.md",
)

print("✅ 已创建 2 个 Tasks\n")

# ============================================================
# 创建 Crew（⭐ 修改配置）
# ============================================================

crew = Crew(
    agents=[joke_researcher, joke_writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    memory=False,  # ⭐ 改为 False（避免 Embeddings 错误）
    cache=False,   # ⭐ 改为 False（简化）
    max_rpm=100,
    share_crew=True,
    verbose=True,  # 保持 True（可以在 AgentOps 中看到详细日志）
)

print("✅ 已创建 Crew\n")

# ============================================================
# 执行任务
# ============================================================

print("="*60)
print("🚀 开始执行任务...")
print("="*60)
print(f"📋 主题: AI engineer jokes\n")

result = crew.kickoff(inputs={"topic": "AI engineer jokes"})

print("\n" + "="*60)
print("📊 最终结果")
print("="*60)
print(result)
print("\n" + "="*60)
print("✅ 任务完成！")
print("="*60)

# ============================================================
# AgentOps 会话信息
# ============================================================

print("\n💡 提示:")
print("   - 访问 AgentOps 控制台查看详细执行追踪")
print("   - 会话数据已自动上传到 AgentOps")
print()