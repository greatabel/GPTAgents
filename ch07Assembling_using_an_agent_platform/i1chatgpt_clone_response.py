import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import os

# streamlit run i1chatgpt_clone_response.py

load_dotenv()

# ===== 本地 OpenAI API 兼容配置 =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

st.title("Local LLM Chat (OpenAI API Compatible)")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = OPENAI_MODEL

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 渲染历史消息
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 输入框
if prompt := st.chat_input("What do you need?"):
    st.session_state["messages"].append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("The assistant is thinking..."):
        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=st.session_state["messages"],
                temperature=0.7,
                max_tokens=2048,
            )

            content = response.choices[0].message.content
            st.markdown(content)

    st.session_state["messages"].append(
        {"role": "assistant", "content": content}
    )
