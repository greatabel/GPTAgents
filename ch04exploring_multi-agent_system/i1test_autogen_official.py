# test_autogen_official.py
"""
使用官方 AutoGen (pyautogen) 测试 DeepSeek 工具调用

前提：
1. pip install pyautogen
2. 代理服务器运行: python deepseek_proxy_server.py
3. OAI_CONFIG_LIST 文件存在
"""

from autogen import ConversableAgent, UserProxyAgent, config_list_from_json

print("="*60)
print("🧪 测试官方 AutoGen with DeepSeek")
print("="*60)
print()

# 加载配置文件
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

print(f"✅ 加载了 {len(config_list)} 个配置:")
for i, cfg in enumerate(config_list, 1):
    print(f"   {i}. 模型: {cfg['model']}, URL: {cfg.get('base_url', 'default')}")
print()

# 创建使用 LLM 的智能体
assistant = ConversableAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)

# 创建代表用户的智能体
user_proxy = UserProxyAgent(
    name="user",
    code_execution_config={
        "work_dir": "working",
        "use_docker": False,
    },
    human_input_mode="ALWAYS",  # 总是等待用户输入
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

print("🚀 开始对话...")
print()

# 开始对话
user_proxy.initiate_chat(
    assistant, 
    message="请介绍一下 Python 的主要特点。回答完后说 TERMINATE。"
)

print()
print("="*60)
print("✅ 对话完成")
print("="*60)