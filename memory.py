"""In-process short-term conversation memory.

A local LLM call is stateless: Ollama does not remember earlier turns unless
we send them again. This class keeps those turns in a Python list on the
current process. That list is the agent's short-term memory.

It is not a database, not embeddings, and not long-term memory. If the
program exits, the list is gone.
"""

from typing import List, TypedDict


class Message(TypedDict):
    """One turn in the conversation."""

    role: str
    content: str


class ConversationMemory:
    """Store user and assistant messages for the current chat session."""

    def __init__(self) -> None:
        self._messages: List[Message] = []

    def add_user_message(self, content: str) -> None:
        """Remember what the user said."""
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Remember what the model said."""
        self._messages.append({"role": "assistant", "content": content})

    def get_history(self) -> List[Message]:
        """Return a copy of the conversation so callers cannot mutate internals."""
        return list(self._messages)

    def clear(self) -> None:
        """Forget every stored turn."""
        self._messages.clear()

    def format_for_prompt(self) -> str:
        """Turn the list into one string for Ollama's /api/generate endpoint.

        /api/generate takes a single `prompt` field, not a chat messages array.
        Short-term memory is useful only if we paste it back into that prompt.
        """
        lines = []
        for message in self._messages:
            label = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"{label}: {message['content']}")
        return "\n".join(lines)
