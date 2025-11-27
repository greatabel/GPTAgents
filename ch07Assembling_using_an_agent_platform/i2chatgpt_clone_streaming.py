import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import os

# streamlit run i2chatgpt_clone_streaming.py

load_dotenv()

st.title("ChatGPT-like clone (Local LLM)")

# ===== 本地 LLM 配置（OpenAI API 兼容） =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")  # 本地一般不校验，给个占位即可
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

# ===== 会话状态 =====
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = OPENAI_MODEL_NAME  # 用本地模型名

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 展示历史消息
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 输入 & 流式回复
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # OpenAI 兼容本地 LLM 的流式输出
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # st.write_stream 支持传一个生成器或可迭代对象
        def stream_text():
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        response = st.write_stream(stream_text)

    st.session_state.messages.append({"role": "assistant", "content": response})
