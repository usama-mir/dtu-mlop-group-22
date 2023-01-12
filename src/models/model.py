# DistilBERT

from transformers import DistilBertForSequenceClassification
# import torch.nn as nn
import torch

Distil_bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

Distil_bert.classifier = torch.nn.Sequential(
                         torch.nn.Linear(768,7),
                         torch.nn.Sigmoid())


