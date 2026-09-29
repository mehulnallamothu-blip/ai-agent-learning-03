# Short-term memory for a local AI agent

A small Python project for learning how conversation memory works inside an agent. It talks to Ollama `llama3.2` over HTTP. There is no OpenAI client, no database, and no agent framework (no LangChain, LangGraph, or CrewAI).

The core idea: **the model does not remember previous turns by itself.** Each `/api/generate` call is independent. Short-term memory is a Python list that we keep in the process and paste back into the next prompt.

## Prerequisites

1. Install [Ollama](https://ollama.com).
2. Start Ollama.
3. Pull the model:

   ```sh
   ollama pull llama3.2
   ```

## Run it

Create and activate a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with `.\.venv\Scripts\Activate.ps1`.

Install dependencies (none beyond the standard library):

```sh
python -m pip install -r requirements.txt
```

Ask llama3.2 a question without memory:

```sh
python local_llm.py
```

Chat with short-term memory:

```sh
python agent.py
```

In the chat:

- Type a message to talk to the model.
- `/history` prints the Python list of stored turns.
- `/clear` empties that list. After this, the model should no longer know earlier facts.
- `/quit` exits.

Try telling it your name, then asking "What is my name?" Then run `/clear` and ask again.

## How memory works internally

1. `ConversationMemory` holds a list of `{role, content}` dicts.
2. The user message is appended to that list.
3. `format_for_prompt()` turns the list into one string.
4. That string is sent to `http://127.0.0.1:11434/api/generate`.
5. The assistant reply is appended to the same list.

If you skip step 3, the model has no history. That is why "memory" here is just **prompt stuffing**: we resend the conversation every turn.

This is short-term (session) memory only. It lives in RAM. It is not embeddings, not a vector store, and not long-term memory.

## Files

- `agent.py` — a simple chat loop that reads and writes `ConversationMemory`.
- `local_llm.py` — POSTs a prompt to `http://127.0.0.1:11434/api/generate` using model `llama3.2`.
- `memory.py` — store user messages, store assistant messages, return history, clear history.
- `requirements.txt` — empty of third-party packages; the Ollama client uses `urllib`.
- `.gitignore` — keeps virtual environments, env files, and Python cache out of version control.
