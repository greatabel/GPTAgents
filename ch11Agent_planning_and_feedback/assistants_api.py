import os
import openai
from dotenv import load_dotenv

load_dotenv()


class AssistantsAPI:
    """
    这里不再使用 OpenAI Assistants / Threads API，
    而是包装成一个简单的 Chat Completions API。
    """

    def __init__(self):
        # 使用本地 OpenAI 兼容接口
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        )
        # 可以用环境变量配置模型名
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # 下面这些 Assistants 相关方法保留空壳，为了让其它文件还能 import / 调用，
    # 但不再真正访问 /v1/assistants 或 /v1/threads。
    def create_thread(self):
        # 不再真正创建远程 thread，仅返回一个占位对象
        class DummyThread:
            id = "local-thread"
        return DummyThread()

    def create_thread_message(self, thread_id, role, content):
        # 用 chat.completions 时，我们不在这里发消息
        return None

    def create_assistant(
        self,
        name,
        instructions,
        model,
        tools,
        files,
        response_format,
        temperature,
        top_p,
    ):
        # 本地后端一般不支持 Assistants 概念，这里直接返回一个占位对象
        class DummyAssistant:
            def __init__(self, _id, _name, _instructions, _model):
                self.id = _id
                self.name = _name
                self.instructions = _instructions
                self.model = _model

        return DummyAssistant(
            _id="local-assistant",
            _name=name,
            _instructions=instructions,
            _model=model or self.model,
        )

    def run_stream(self, thread_id, assistant_id, event_handler):
        """
        这个接口在原来是 threads.runs.stream，这里我们不用它，
        真正的流式在 gradio_assistants_chatbot.run() 里直接用 chat.completions。
        但为了兼容，不在这里删除。
        """
        raise NotImplementedError("run_stream is not used in local chat-completions mode.")

    def list_assistants(self):
        # 返回一个空列表结构，给 gradio_assistants_panel 用
        class DummyList:
            data = []
        return DummyList()

    def retrieve_assistant(self, assistant_id):
        # 简单返回一个占位 assistant，主要是让 UI 还能正常运行
        class DummyAssistant:
            def __init__(self, _id, _name, _instructions, _model):
                self.id = _id
                self.name = _name
                self.instructions = _instructions
                self.model = _model

        return DummyAssistant(
            _id=assistant_id or "local-assistant",
            _name="Local Assistant",
            _instructions="You are a helpful assistant.",
            _model=self.model,
        )

    def update_assistant(
        self,
        assistant_name,
        assistant_id,
        assistant_instructions,
        assistant_model,
        assistant_tools,
        assistant_files,
        assistant_resformat,
        assistant_temperature,
        assistant_top_p,
    ):
        # 简单返回新的 DummyAssistant
        class DummyAssistant:
            def __init__(self, _id, _name, _instructions, _model):
                self.id = _id
                self.name = _name
                self.instructions = _instructions
                self.model = _model

        return DummyAssistant(
            _id=assistant_id or "local-assistant",
            _name=assistant_name,
            _instructions=assistant_instructions,
            _model=assistant_model or self.model,
        )

    def delete_assistant(self, assistant_id):
        # 本地模式下什么也不做
        return None

    # —— 新增：真正给外界用的 chat.completions 流式接口 ——
    def stream_chat(self, messages):
        """
        messages: [{'role': 'user'/'assistant', 'content': '...'}, ...]
        """
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )


api = AssistantsAPI()