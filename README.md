# AI Agent — Calculator, Tasks & RAG Assistant

A command-line chat agent that can do arithmetic, look up your to-do list, and answer questions about reinforcement learning by searching your own notes. It uses Claude (via the Anthropic API) to decide which tool to use for a given question, and Voyage AI to power the notes search (RAG).

## Features

- **Arithmetic** — ask any math question and it's evaluated exactly, not guessed by the model.
- **Task list lookup** — ask about your pending/completed to-dos (reads from `tasks.json`).
- **Notes search (RAG)** — ask conceptual questions about reinforcement learning, and the agent searches a local notes file (`rl_primer.txt`) for the most relevant section before answering. If the notes don't cover something, it says so explicitly rather than making it up.
- **Resilient to transient failures** — API calls automatically retry with a backoff delay if they hit a rate limit or temporary error.
- **Works without the notes file** — if `rl_primer.txt` is missing, the calculator and task tools still work; only notes search is disabled.

## Requirements

- Python 3.10 or later
- An [Anthropic API key](https://console.anthropic.com/)
- A [Voyage AI API key](https://dashboard.voyageai.com/) (used for embedding your notes; the free tier is enough for this project)

## Setup

1. Clone this repository and move into the project folder.

2. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Create a file named `.env` in the project folder with your API keys:

   ```
   ANTHROPIC_API_KEY=your-anthropic-key-here
   VOYAGE_API_KEY=your-voyage-key-here
   ```

   These are the exact variable names the Anthropic and Voyage AI SDKs look for automatically — don't rename them.

4. Make sure `tasks.json` and `rl_primer.txt` exist in the project folder (see **Project files** below). If `rl_primer.txt` is missing, the agent still runs — notes search just won't be available.

## Usage

Run the agent from the project folder:

```
python agent.py
```

You'll get an interactive prompt. Type a question and press enter; type `quit` to exit.

Example session:

```
You: what is 522 * 785?
Agent: 522 * 785 = 409,770

You: how many tasks are pending?
Agent: You have 3 pending tasks: T1, T3, and T4.

You: what is reinforcement learning?
Agent: Based on your notes: RL is a branch of machine learning where an agent
learns to make decisions by interacting with an environment and receiving
feedback in the form of rewards...
```

## Project files

- `agent.py` — the agent itself: tool definitions, the retry/RAG logic, and the interactive chat loop.
- `tasks.json` — your task list, read by the `get_tasks` tool. Format: a JSON array of `{"task": "...", "completed": true/false}` objects.
- `rl_primer.txt` — your notes file, read and embedded for the `search_documents` tool. Paragraphs separated by a blank line are treated as individual searchable chunks.
- `doc_embeddings_cache.json` — auto-generated the first time the agent runs; caches your notes' embeddings so they aren't recomputed (and re-billed) on every run. Safe to delete — it will be rebuilt automatically.
- `requirements.txt` — the third-party Python packages this project depends on.

## Running the evaluation suite

`eval_agent.py` is an automated test harness that checks the agent's behavior against a fixed set of questions and expected answers (some checked by exact match, some graded by an LLM judge for open-ended questions). Run it after making any change to `agent.py` to confirm nothing broke:

```
python eval_agent.py
```

It prints a pass/fail result for each test case and an overall success rate.
