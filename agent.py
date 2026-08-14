import os
from dotenv import load_dotenv
import anthropic
import json 

load_dotenv()
client = anthropic.Anthropic()

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

    
