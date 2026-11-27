# Week 7 – Next-Token Probabilities Using Hugging Face Transformers

## Objective
To inspect next-token predictions from a small open-source language model by examining raw logits, calculating probabilities, and displaying the Top-K candidate tokens.

## Tools and Technologies
- Python
- Hugging Face Transformers
- PyTorch
- DistilGPT-2

## What I Implemented
1. Loaded the DistilGPT-2 language model.
2. Accepted a text prompt from the user.
3. Converted the prompt into tokens using the tokenizer.
4. Extracted the raw logits for the next token.
5. Applied the Softmax function to convert logits into probabilities.
6. Displayed the Top-5 most probable next-token candidates.
7. Displayed the raw logits and probabilities of each candidate.

## What I Learned
- A language model predicts the next token based on the input context.
- Logits are raw scores produced by the model.
- Softmax converts logits into probability values.
- Top-K selects the K most probable token candidates.
- Different prompts can produce different next-token predictions.

## Model Used
DistilGPT-2

## Execution
Run the following command:

```bash
python main.py
```

Enter a text prompt when requested.

## Result
The experiment successfully displayed the Top-5 next-token candidates, their probability percentages, and their raw logits.

**Status: Successfully Completed**