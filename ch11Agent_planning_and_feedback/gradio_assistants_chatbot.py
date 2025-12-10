import re
import gradio as gr

from assistants_api import api  # 已改成本地 chat.completions 封装
from assistants_utils import EventHandler
from gradio_assistants_panel import assistants_panel

# dummy thread，占位避免其它代码引用报错
thread = api.create_thread()


def wrap_latex_with_markdown(text: str) -> str:
    bracket_pattern = re.compile(r"\[(.*?)\]")
    parenthesis_pattern = re.compile(r"\((.*?)\)")
    text = bracket_pattern.sub(r"$$\1$$", text)
    text = parenthesis_pattern.sub(r"$$\1$$", text)
    return text


def respond(user_input, history, assistant_id):
    """
    使用“messages 模式”的 Chatbot：
    - history: List[{"role": "user"|"assistant", "content": "..."}]
    """
    if history is None:
        history = []

    user_input = (user_input or "").strip()
    if not user_input:
        return history, ""  # 空输入，直接返回

    # 把当前用户消息加到 messages 列表
    history.append({"role": "user", "content": user_input})

    # 取 assistant（目前只是占位）
    _assistant = api.retrieve_assistant(assistant_id)
    _eh = EventHandler([])

    # 用 chat.completions 流式接口，在这里一次性收完
    full_text = ""
    with api.stream_chat(history) as stream:
        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                full_text += delta.content

    full_text = wrap_latex_with_markdown(full_text)

    # 把 assistant 回复也加入 messages 列表
    history.append({"role": "assistant", "content": full_text})

    # 返回：更新后的 history，清空输入框
    return history, ""


custom_css = """
:root {
    --adjustment-ratio: 150px; /* Height to subtract from the viewport for chatbot */
}

body, html {
    height: 100%;
    width: 100%;
    margin: 0;
    padding: 0;
}

.gradio-container {
    max-width: 100% !important; 
    width: 100%;
}

#chatbot {
    height: calc(100vh - var(--adjustment-ratio)) !important;
    overflow-y: auto !important;
}

#instructions textarea {
    min-height: calc(100vh - (var(--adjustment-ratio) + 750px));
    max-height: 1000px;
    resize: vertical;
    overflow-y: auto;
}
#assistant_logs textarea {
    height: calc(100vh - (var(--adjustment-ratio) - 25px));
    min-height: 150px;
    max-height: 1000px;
    resize: vertical;
    overflow-y: auto;
    box-sizing: border-box;
}
"""

with gr.Blocks() as demo:
    with gr.Row():
        # 左侧：assistant 选择面板
        with gr.Column(scale=2):
            assistant_id = assistants_panel()

        # 中间：Chatbot + Textbox，Chatbot 使用 messages 结构
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                value=[],
                elem_id="chatbot",
                label="Chatbot",
            )
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter message...",
            )
            # history 直接是 messages list
            state = gr.State([])

            submit_event = txt.submit(
                fn=respond,
                inputs=[txt, state, assistant_id],
                outputs=[chatbot, txt],
                queue=True,
            )
            # 把 Chatbot 的 messages 回写到 state
            submit_event.then(
                lambda messages: messages,
                inputs=chatbot,
                outputs=state,
            )

        # 右侧：日志区域（目前只是占位）
        with gr.Column(scale=2):
            assistant_logs = gr.Markdown(
                "Assistant Logs", elem_id="assistant_logs", render=False
            )
            assistant_logs.render()

if __name__ == "__main__":
    demo.queue()
    demo.launch(css=custom_css)