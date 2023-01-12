from transformers import DistilBertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np 

# Using Pretrained DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Creating Dataset class for Toxic comments and Labels 
class Toxic_Dataset(Dataset):
    def __init__(self, Comments_, Labels_):
        self.comments = Comments_.copy()
        self.labels = Labels_.copy()
        
        self.comments["comment_text"] = self.comments["comment_text"].map(lambda x: tokenizer(x, padding="max_length", truncation=True, return_tensors="pt"))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        comment = self.comments.loc[idx,"comment_text"]
        label = np.array(self.labels.loc[idx,:])
        
        return comment, label


