# snake_game_with_cache_fixed.py
"""
Snake 游戏开发 - 带缓存的修复版
"""

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.cache import Cache

print("="*60)
print("🎮 demo(Fixed)")
print("="*60)
print(f"📅 Date: 2025-11-19 15:27:05 UTC")
print(f"👤 User: greatabel")
print()

# Load the configuration list from the config file.
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

print(f"✅ Config loaded: {config_list[0]['model']}")
print()

# Create the agent that represents the user in the conversation.
user_proxy = UserProxyAgent(
    "user",
    code_execution_config={
        "work_dir": "working",
        "use_docker": False,
        "last_n_messages": 1,
    },
    human_input_mode="TERMINATE",  # ⭐ 修改 1
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    max_consecutive_auto_reply=10,  # ⭐ 修改 2
)

engineer = AssistantAgent(
    name="Engineer",
    llm_config={"config_list": config_list},
    system_message="""
    You are a profession Python engineer, known for your expertise in software development.
    You use your skills to create software applications, tools, and games that are both functional and efficient.
    Your preference is to write clean, well-structured code that is easy to read and maintain.    
    """,
)

critic = AssistantAgent(
    name="Reviewer",
    llm_config={"config_list": config_list},
    system_message="""
    You are a code reviewer, known for your thoroughness and commitment to standards.
    Your task is to scrutinize code content for any harmful or substandard elements.
    You ensure that the code is secure, efficient, and adheres to best practices.
    You will identify any issues or areas for improvement in the code and output them as a list.
    """,
)


def review_code(recipient, messages, sender, config):
    return f"""
            Review and critque the following code.
            
            {recipient.chat_messages_for_summary(sender)[-1]['content']}
            """


user_proxy.register_nested_chats(
    [
        {
            "recipient": critic,
            "message": review_code,
            "summary_method": "last_msg",
            "max_turns": 3,
        }
    ],
    trigger=engineer,
)

task = """Write a small demo using python."""

print("🗄️  Using disk cache (seed=42)...")
print("💡 Tip: Run again to see cache in action!")
print()

with Cache.disk(cache_seed=42) as cache:
    res = user_proxy.initiate_chat(
        recipient=engineer,
        message=task,
        max_turns=5,  # ⭐ 修改 3: 2 → 5
        summary_method="last_msg",
        cache=cache,
    )

print()
print("="*60)
print("📊 Conversation Summary")
print("="*60)
print(res.summary[:200] + "..." if len(res.summary) > 200 else res.summary)
print()