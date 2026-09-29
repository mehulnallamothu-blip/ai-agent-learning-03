"""A tiny chat loop that shows how short-term memory works.

There is no tool calling and no planning. Each turn:

1. Store the user message in a Python list.
2. Send the whole list to llama3.2 as one prompt.
3. Store the assistant reply in the same list.

The model only "remembers" earlier turns because we resend them.
"""

from local_llm import ask_local_llm
from memory import ConversationMemory

INSTRUCTIONS = """
You are a helpful assistant. Use the conversation history below to stay
consistent. If the user told you their name or a fact earlier, use it.
If there is no history yet, just answer the latest message.
""".strip()


def build_prompt(memory: ConversationMemory) -> str:
    """Build one prompt from instructions plus every stored turn."""
    history = memory.format_for_prompt()
    if history:
        return f"{INSTRUCTIONS}\n\nConversation so far:\n{history}\n\nAssistant:"
    return f"{INSTRUCTIONS}\n\nConversation so far:\n(empty)\n\nAssistant:"


def main() -> None:
    memory = ConversationMemory()
    print("Chat with llama3.2. Commands: /clear  /history  /quit")

    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue

        if user_text == "/quit":
            return

        if user_text == "/clear":
            memory.clear()
            print("Memory cleared. The model will not see earlier turns.")
            continue

        if user_text == "/history":
            history = memory.get_history()
            if not history:
                print("Memory is empty.")
            else:
                for message in history:
                    print(f"{message['role']}: {message['content']}")
            continue

        memory.add_user_message(user_text)
        reply = ask_local_llm(build_prompt(memory))
        memory.add_assistant_message(reply)
        print("\nAssistant:")
        print(reply)


if __name__ == "__main__":
    main()
