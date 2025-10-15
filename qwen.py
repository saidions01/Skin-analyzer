from ollama import Client
# Create a local client (default: http://localhost:11434)
client = Client()

# Send a simple text prompt to the mistral model
response = client.generate(
    model="qwen3:1.7b",        # model name (the one you pulled)
    prompt="Write a short haiku about autumn winds."
)

# Print the model’s response text
print(response["response"])