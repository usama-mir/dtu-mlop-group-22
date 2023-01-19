from fastapi import FastAPI, Response
import torch
from transformers import DistilBertTokenizer
import numpy as np

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


model = torch.load("../models/model_epoch2.pth")
model.eval()
classes = [
    "toxic, severe_toxic, obscene",
    "threat",
    "insult",
    "identity_hate",
    "non_toxic",
]

# Defining the prediction endpoint without data validation
@app.get("/predict")
async def basic_predict(comment: str):

    # Getting the JSON from the body of the request

    tk = tokenizer(comment, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = tk["input_ids"]
    attention_masks = tk["attention_mask"]

    ids = torch.tensor(input_ids, dtype=torch.long)
    mask = torch.tensor(attention_masks, dtype=torch.long)

    output = model(input_ids=ids.reshape(-1, 1), attention_mask=mask)
    output = output.logits.detach()

    predicted_classes = np.array(torch.argmax(output, dim=0))
    actual_classes = np.where(predicted_classes == 1)[0]
    final_class_labels = []
    for i in actual_classes:
        final_class_labels.append(classes[i])

    return Response(content=str(final_class_labels))
