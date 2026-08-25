from agent import ask_agent, get_tasks, calculate, search_documents, cosine_similarity, anthropic, voyageai

client = anthropic.Anthropic()


messages_list = [{"input": "what is 522 * 785", "check": ["409,770"]},
 {"input": "how many tasks are pending?", "check": ["3"]},
 {"input": "what is RL?",
  "check": {"judge": "The reply should explain that reinforcement learning is a machine learning approach where an agent learns through trial and error by taking actions, receiving reward feedback, and interacting with an environment."}},
 {"input": "add the number of pending tasks and number of completed tasks", "check": ["4"]},
 {"input": "what is square of a table?",
 "check": {"judge": "The reply should explain that the square of a table is not a standard mathematical concept, and ask the user for clarification if any part of question was incorrectly phrased."}},
 {"input": "how does neural networks work?",
 "check": {"judge": "The reply should mention that neural networks are not detailed in the notes provided, and assistant can ask if the user needs  information from external sources."}}
  ]

def llm_judge(reply, criteria):
    judge_prompt = f"""You are evaluating whether an AI assistant's reply satisfies a criterion.
        Reply: {reply}
        Criterion: {criteria}
        Does the reply satisfy the criterion? Answer with exactly one word: PASS or FAIL."""
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    verdict = response.content[0].text.strip().upper()
    return verdict.startswith("PASS")

counter = []

for test in messages_list:
    query = test["input"]
    message = []

    print(f"-------------------------------------------------------------")
    print(f"Testing input: {query}")
    reply = ask_agent(query, message).lower()
    print(f"Agent: {reply}")

    if "check" in test:
        check = test["check"]
        if isinstance(check, dict) and "any" in check:
            if any(x in reply for x in check["any"]):
                print("Test passed (any match).")
                counter.append(1)
            else:
                print("Test failed (no match).")
                counter.append(0)                
        elif isinstance(check, list):
            if all(x in reply for x in check):
                print("Test passed (all match).")
                counter.append(1)
            else:
                print("Test failed (not all match).")
                counter.append(0)
        elif isinstance(check, dict) and "judge" in check:
            if llm_judge(reply, check["judge"]):
                print("Test passed (LLM judge).")
                counter.append(1)
            else:
                print("Test failed (LLM judge).")
                counter.append(0)

print(f"Passed {sum(counter)} out of {len(messages_list)} tests. Success rate: {sum(counter)/len(messages_list)*100:.2f}%")
