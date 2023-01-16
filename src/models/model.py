# DistilBERT

from transformers import DistilBertForSequenceClassification
import torch.nn as nn

Distil_bert = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"
)

Distil_bert.classifier = nn.Sequential(nn.Linear(768, 7), nn.Sigmoid())
