# snake_game_minimal_fix.py
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Load the configuration list from the config file.
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

# Create the agent that represents the user in the conversation.
user_proxy = UserProxyAgent(
    "user",
    code_execution_config={
        "work_dir": "working",
        "use_docker": False,
        "last_n_messages": 1,
    },
    human_input_mode="TERMINATE",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    max_consecutive_auto_reply=10,
)

# ----------- Assistant 1：总结助手 -----------
engineer = AssistantAgent(
    name="Summarizer",
    llm_config={"config_list": config_list},
    system_message="""
    你是一个专门生成段落总结的助手。
    你的任务是：阅读用户给出的段落，并生成一个简洁、准确的总结。
    """,
)

# ----------- Assistant 2：评价助手 -----------
critic = AssistantAgent(
    name="Reviewer",
    llm_config={"config_list": config_list},
    system_message="""
    你是一个评价文本质量的助手。
    你的任务是：评价总结的清晰度、语言质量、简洁性与语法正确性。
    """,
)

# Reviewer 的输入信息构造
def review_code(recipient, messages, sender, config):
    return f"""
    请对下面这段由 Summarizer 生成的总结进行评价：

    {recipient.chat_messages_for_summary(sender)[-1]['content']}
    """

# 注册：让 Reviewer 在 Summarizer 完成内容后自动接力
user_proxy.register_nested_chats(
    [
        {
            "recipient": critic,
            "message": review_code,
            "summary_method": "last_msg",
            "max_turns": 1,
        }
    ],
    trigger=engineer,
)

# ---------------- 用户任务 ----------------
task = """
请总结下面这段话：

太阳是一颗恒星，是太阳系的中心天体。
它通过核聚变释放巨大的能量，为地球提供光和热，
并维持地球上的生命活动。
"""

# 启动对话
res = user_proxy.initiate_chat(
    recipient=engineer,
    message=task,
    max_turns=1,
    summary_method="last_msg"
)
