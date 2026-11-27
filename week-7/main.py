
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Load the small open-source model
MODEL_NAME = "distilgpt2"

print("=" * 60)
print("       NEXT-TOKEN PROBABILITY EXPERIMENT")
print("=" * 60)

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

print("Model loaded successfully!")

# Step 2: Take input from the user
prompt = input("\nEnter your text prompt: ")

# Step 3: Convert text into tokens
inputs = tokenizer(prompt, return_tensors="pt")

# Step 4: Get model output and logits
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

# Get logits for the next token
next_token_logits = logits[0, -1, :]

# Step 5: Convert logits into probabilities
probabilities = torch.softmax(next_token_logits, dim=-1)

# Step 6: Find Top 5 next-token candidates
top_k = 5

top_probabilities, top_indices = torch.topk(probabilities, top_k)

# Step 7: Display results
print("\n" + "=" * 60)
print("             NEXT TOKEN PREDICTIONS")
print("=" * 60)

print("\nInput Prompt:", prompt)

print("\nTop 5 Next-Token Candidates:\n")

for rank, (prob, token_id) in enumerate(
    zip(top_probabilities, top_indices), start=1
):
    token = tokenizer.decode([token_id.item()])

    print(
        f"{rank}. Token: {repr(token):15} "
        f"Probability: {prob.item() * 100:.4f}%"
    )

# Step 8: Display raw logits
print("\nRaw Logits of Top 5 Candidates:")

for token_id in top_indices:
    token = tokenizer.decode([token_id.item()])
    logit = next_token_logits[token_id].item()

    print(f"Token: {repr(token):15} Logit: {logit:.4f}")

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED SUCCESSFULLY!")
print("=" * 60)
