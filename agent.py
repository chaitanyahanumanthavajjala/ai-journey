import os
import anthropic
import json 
import voyageai
import math
import time


from dotenv import load_dotenv
load_dotenv()

notes_filename = "rl_primer.txt"

SYSTEM_PROMPT = """You have access to three tools: calculate, get_tasks, and search_documents.
Always use search_documents to check the user's notes before answering any conceptual or
informational question from your own general knowledge, even if you're confident you already
know the answer. If the notes don't contain relevant information, say so explicitly rather than
silently answering from your own knowledge instead.
Always use calculate for arithmetic instead of computing it yourself.
Always use get_tasks when asked about tasks or to-dos."""

anthropic_retries = 3
anthropic_retry_delay = 3

voyageai_retries = 3
voyageai_retry_delay = 30


def call_with_retry(fn, max_retries=3, delay_seconds=30):
    # fn is a function that takes no arguments — call it, and if it raises,
    # wait delay_seconds and try again, up to max_retries total attempts.
    # If every attempt fails, let the last exception propagate.
    for _ in range(max_retries):
        try:
            return fn()
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in {delay_seconds} seconds...")
            if _ < max_retries - 1:
                time.sleep(delay_seconds)
    raise Exception("All retry attempts failed.")


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x**2 for x in a))
    magnitude_b = math.sqrt(sum(x**2 for x in b))
    return dot_product / (magnitude_a * magnitude_b)

try:
    with open(notes_filename, "r") as f:
        text = f.read()
    doc_chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    #for i, chunk in enumerate(doc_chunks):
    #   print(f"Chunk {i}: {chunk[:60]}...")

except FileNotFoundError:
    doc_chunks = []
    print(f"{notes_filename} not found. Document search will not work without it.")


client = anthropic.Anthropic()
vo = voyageai.Client()

if doc_chunks != []:
    # ONE-TIME SETUP — runs once, before the loop
    # ... open file, chunk, embed all chunks (vo.embed(chunks, ...)) ...
    CACHE_FILE = "doc_embeddings_cache.json"
    # doc_chunks gets built the same way as always, from rl_primer.txt — that part's free, no API call
    # then, instead of unconditionally calling vo.embed(...):
    cached = None
    try:
        with open(CACHE_FILE, "r") as f:
            cached = json.load(f)
    except FileNotFoundError:
        pass

    if cached and cached["chunks"] == doc_chunks:
        doc_embeddings = cached["embeddings"]
    else:
        doc_result = call_with_retry(lambda: vo.embed(doc_chunks, model="voyage-4", input_type="document"), max_retries=voyageai_retries, delay_seconds=voyageai_retry_delay)
        doc_embeddings = doc_result.embeddings
        with open(CACHE_FILE, "w") as f:
            json.dump({"chunks": doc_chunks, "embeddings": doc_embeddings}, f)
else:
    doc_embeddings = []


#for i, embedding in enumerate(doc_result.embeddings):
#   print(f"Chunk {i}: {len(embedding)} numbers, starts with {embedding[:5]}")

tools = [
    {
        "name": "calculate",
        "description": "Evaluate a basic math expression and return the numeric result. Use this for any arithmetic instead of computing it yourself.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression to evaluate, e.g. '847 * 92'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_tasks",
        "description": "Get the user's current to-do list, including which tasks are done and which are pending. Use this whenever the user asks about their tasks, to-dos, or what they still need to do.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
            }
    },
    {
        "name": "search_documents",
        "description": "Search the user's personal notes/documents for relevant information. Use this whenever the user asks about concepts that might be explained in their notes — for example, questions about reinforcement learning topics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A search query to find relevant information in the documents."
                }
            },
            "required": ["query"]
        }
    }
]

def get_tasks():
    try:
        with open("tasks.json", "r") as f:
            tasks = json.load(f)
    except FileNotFoundError:
        return "No tasks file found."
    if not tasks:
        return "No tasks currently."
    lines = [f"{t['task']} - {'done' if t['completed'] else 'pending'}" for t in tasks]
    return "\n".join(lines)

def calculate(expression):
    return str(eval(expression))  # eval() flagged earlier as unsafe for real products — fine here

def search_documents(query):
    if doc_embeddings == []:
        return "No notes are available right now."
    else: 
        query_embedding = call_with_retry(lambda: vo.embed([query], model="voyage-4", input_type="query").embeddings[0], max_retries=voyageai_retries, delay_seconds=voyageai_retry_delay)
        scores = [(i, cosine_similarity(query_embedding, emb)) for i, emb in enumerate(doc_embeddings)]
        best_index, best_score = max(scores, key=lambda x: x[1])
        return doc_chunks[best_index]

def ask_agent(user_input, messages):
    messages.append({"role": "user", "content": user_input})

    MAX_ITERATIONS = 5
    iteration = 0

    while True:
        iteration += 1
        #print(f"Iteration: {iteration} for user input: {user_input}")
        if iteration > MAX_ITERATIONS:
            print("GUARDRAIL TRIGGERED: Too many tool calls without a final answer. Forcing a text response.")
            messages.append({
                "role": "user",
                "content": "You've made several tool calls without finishing. Please answer now with your best response based on what you know, without calling any more tools."
            })
            
            response = call_with_retry(lambda: client.messages.create(model="claude-sonnet-4-5",
                max_tokens=500,
                messages=messages,
                system=SYSTEM_PROMPT  # no tools= here — forces a text answer
                ), max_retries=anthropic_retries, delay_seconds=anthropic_retry_delay)
            messages.append({"role": "assistant", "content": response.content})
            break

        response = call_with_retry(lambda: client.messages.create(model="claude-sonnet-4-5",
            max_tokens=500,
            tools=tools,
            system=SYSTEM_PROMPT,
            messages=messages
            ), max_retries=anthropic_retries, delay_seconds=anthropic_retry_delay)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break
        print(f"Tool use detected for query: {user_input}. Processing tool calls...")

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    if block.name == "calculate":
                        result = calculate(block.input["expression"])
                    elif block.name == "get_tasks":
                        result = get_tasks()
                    elif block.name == "search_documents":
                        result = search_documents(block.input["query"])
                    else:
                        result = f"Unknown tool: {block.name}"
                except Exception as e:
                    print(f"Tool failed with exception: {e}")
                    result = f"Tool failed: {e}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        messages.append({"role": "user", "content": tool_results})
        # no break here — let the while loop go around again


    final_text = "".join(block.text for block in response.content if block.type == "text")
    return final_text

def main():
    messages = []
    print("Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break    
        reply = ask_agent(user_input, messages)
        print(f"Agent: {reply}")

if __name__ == "__main__":
    main()
    
