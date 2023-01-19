from transformers import DistilBertForSequenceClassification
from transformers import pipeline
import torch.nn as nn 

model_path = 'models/transformers/' # will be created automatically if not exists
model_name = "distillbert"

Distil_bert = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"
)
Distil_bert.classifier = nn.Sequential(nn.Linear(768, 7), nn.Sigmoid())
Distil_bert.save_pretrained(model_path)

