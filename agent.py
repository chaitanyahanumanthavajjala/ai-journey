import os
import anthropic
import json 
import voyageai
import math


from dotenv import load_dotenv
load_dotenv()

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x**2 for x in a))
    magnitude_b = math.sqrt(sum(x**2 for x in b))
    return dot_product / (magnitude_a * magnitude_b)

with open("rl_primer.txt", "r") as f:
    text = f.read()

doc_chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
for i, chunk in enumerate(doc_chunks):
    print(f"Chunk {i}: {chunk[:60]}...")

client = anthropic.Anthropic()
vo = voyageai.Client()
doc_result = vo.embed(doc_chunks, model="voyage-4", input_type="document")
doc_embeddings = doc_result.embeddings

for i, embedding in enumerate(doc_result.embeddings):
    print(f"Chunk {i}: {len(embedding)} numbers, starts with {embedding[:5]}")


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
    query_embedding = vo.embed([query], model="voyage-4", input_type="query").embeddings[0]
    scores = [(i, cosine_similarity(query_embedding, emb)) for i, emb in enumerate(doc_embeddings)]
    best_index, best_score = max(scores, key=lambda x: x[1])
    return doc_chunks[best_index]

messages = []

print("Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    messages.append({"role": "user", "content": user_input})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            tools=tools,
            messages=messages
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break  # Claude has a real answer, stop looping

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
                    result = f"Tool failed: {e}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "user", "content": tool_results})

    for block in response.content:
        if block.type == "text":
            print(block.text)

    
