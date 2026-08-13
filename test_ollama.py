import ollama

response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "user", "content": "In one sentence, what is retrieval-augmented generation?"}
    ]
)

print(response["message"]["content"])
