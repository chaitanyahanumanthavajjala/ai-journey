import voyageai
import anthropic
import math


from dotenv import load_dotenv
load_dotenv()

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x**2 for x in a))
    magnitude_b = math.sqrt(sum(x**2 for x in b))
    return dot_product / (magnitude_a * magnitude_b)


# ONE-TIME SETUP — runs once, before the loop
with open("rl_primer.txt", "r") as f:
    text = f.read()

chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk[:60]}...")

# ... open file, chunk, embed all chunks (vo.embed(chunks, ...)) ...

vo = voyageai.Client()
result = vo.embed(chunks, model="voyage-4", input_type="document")

for i, embedding in enumerate(result.embeddings):
    print(f"Chunk {i}: {len(embedding)} numbers, starts with {embedding[:5]}")

# PER-QUESTION WORK — runs every loop iteration
# ... embed the question, compute similarity against the
# precomputed chunk embeddings, find best match, call Claude,
# print the answer ...

client = anthropic.Anthropic()

print("Type 'quit' to exit.")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break

    scores =[]
    question_embedding = vo.embed([question], model="voyage-4", input_type="query").embeddings[0]

    for i, chunk in enumerate(chunks):
        chunk_embedding = result.embeddings[i]
        score = cosine_similarity(question_embedding, chunk_embedding)
        scores.append((i, score))
        print (f"Chunk {i}: similarity score = {score:.4f}")

    best_index, best_score = max(scores, key=lambda x: x[1])
    best_chunk = chunks[best_index]

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system="Answer the user's question using ONLY the provided context. If the context doesn't contain the answer, say you don't know.",
        messages=[
            {"role": "user", "content": f"Context:\n{best_chunk}\n\nQuestion: {question}"}
        ]
    )    

    print(response.content[0].text)









