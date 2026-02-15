from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import math

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")


def sentence_surprisal(text):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

    input_ids = inputs["input_ids"][0]

    total = 0
    for i, token_id in enumerate(input_ids):
        prob = probs[0, i, token_id]
        total += -math.log(prob.item())

    return total / len(input_ids)


if __name__ == "__main__":
    s1 = "This is easy."
    s2 = "Notwithstanding the aforementioned considerations."

    print("Easy:", sentence_surprisal(s1))
    print("Hard:", sentence_surprisal(s2))
