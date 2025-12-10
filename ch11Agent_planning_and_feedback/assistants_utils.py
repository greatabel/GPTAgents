import datetime

import openai
from dotenv import load_dotenv

load_dotenv()

# 如果本地后端不提供 files.content 等能力，这个 client 可能暂时用不到。
client = openai.OpenAI()


def save_binary_response_content(binary_content):
    # Function to get the current timestamp
    def get_timestamp():
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Function to determine the file extension based on the magic numbers (file signatures)
    def get_file_extension(byte_string):
        if byte_string.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        elif byte_string.startswith(b"\xff\xd8\xff\xe0") or byte_string.startswith(
            b"\xff\xd8\xff\xe1"
        ):
            return "jpg"
        elif byte_string.startswith(b"GIF87a") or byte_string.startswith(b"GIF89a"):
            return "gif"
        elif byte_string.startswith(b"%PDF-"):
            return "pdf"
        # Add more signatures as needed
        else:
            return "bin"  # Generic binary file extension

    # Get the file extension
    extension = get_file_extension(binary_content)

    # Create a unique file name using the timestamp
    timestamp = get_timestamp()
    file_name = f"file_{timestamp}.{extension}"

    # Save the content to the file
    with open(file_name, "wb") as file:
        file.write(binary_content)
        print(f"File saved as {file_name}")

    return file_name


class EventHandler:
    """
    简化版 EventHandler，不再继承 AssistantEventHandler。
    这里只是为了保留原来接口，让其它代码引用 self.logs / self.images 不报错。
    """

    def __init__(self, logs) -> None:
        self._logs = logs
        self._images = []

    @property
    def logs(self):
        return self._logs

    @property
    def images(self):
        return self._images

    # 以下方法仅打印/记录日志，避免出错；真正流式逻辑由 chat.completions 负责
    def on_text_created(self, text) -> None:
        print("assistant > ", end="", flush=True)

    def on_text_delta(self, delta, snapshot=None):
        pass

    def on_image_file_done(self, image_file) -> None:
        # 这里暂时不实现具体逻辑，如果本地服务未来支持文件流可以再补
        self._logs += [f"Image file: {image_file}"]
        self._images += [image_file]

    def on_tool_call_created(self, tool_call):
        self._logs += [f"\nassistant > {tool_call}"]

    def on_tool_call_delta(self, delta, snapshot=None):
        pass

    def on_tool_call_done(self, tool_call) -> None:
        pass