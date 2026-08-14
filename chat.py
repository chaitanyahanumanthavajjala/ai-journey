import os
from dotenv import load_dotenv
import anthropic

load_dotenv()  # reads .env and loads ANTHROPIC_API_KEY into the environment

client = anthropic.Anthropic()  # automatically finds ANTHROPIC_API_KEY from the environment

messages = []

SYSTEM_PROMPT = """You are a patient study companion helping someone learn technical concepts from scratch.

When explaining something:
- Use plain language first, technical terms second (define them when you use them)
- Give a concrete example or analogy, not just an abstract definition
- Keep initial explanations short — a few sentences, not an essay
- End with a short question to check understanding, rather than assuming it landed
- Explain why the concept is useful or important, not just what it is

If the user seems confused or asks a follow-up, go deeper and slower rather than repeating the same explanation louder."""

print("Type 'quit' to exit the chat at any time.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    messages.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=messages
    )
    reply = response.content[0].text
    messages.append({"role": "assistant", "content": reply})
    print(f"Agent: {reply}")
