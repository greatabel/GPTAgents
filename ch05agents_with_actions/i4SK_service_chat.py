import asyncio
import os
import semantic_kernel as sk

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments

from plugins.Movies.tmdb import TMDbService


def history_to_str(history):
    return "\n".join(f"{msg.role}: {msg.content}" for msg in history)


async def main():
    kernel = sk.Kernel()

    api_key = os.getenv("OPENAI_API_KEY", "dummy")
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    model_name = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["OPENAI_BASE_URL"] = api_base

    service_id = "local_llm"

    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id=model_name,
            api_key=api_key,
        )
    )

    kernel.add_plugin(TMDbService(), plugin_name="TMDBService")

    history = []
    history.append(("system", "You are a helpful assistant that recommends movies and TV shows."))

    print("Welcome to the TMDB chat bot (local LLM). Type 'exit' to quit.\n")

    while True:
        user_input = input("User:> ").strip()
        if user_input.lower() == "exit":
            break

        history.append(("user", user_input))

        history_str = "\n".join(f"{r}: {c}" for r, c in history)

        result = await kernel.invoke_prompt(
            prompt="{{$chat_history}}",
            arguments=KernelArguments(chat_history=history_str),
            service_id=service_id,
            system_prompt="You recommend movies and TV shows. Use tools if useful.",
            tool_choice="auto",
        )

        assistant_message = str(result)
        history.append(("assistant", assistant_message))

        print(f"Assistant:> {assistant_message}")


if __name__ == "__main__":
    asyncio.run(main())
