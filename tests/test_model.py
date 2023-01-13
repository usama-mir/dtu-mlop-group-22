from src.models.model import Distil_bert
from src.models import train_model
from transformers import DistilBertForSequenceClassification,DistilBertModel, DistilBertTokenizer, DistilBertForTokenClassification, DistilBertConfig
import torch
import torch.nn as nn
import pytest
import sys
import random
import numpy as np

sys.path.insert(0, "src")

# @pytest.mark.parametrize(
#     "ids, mask", [([[101, 1188, 1110, 102, 0, 0, 0]], [[1, 1, 1, 1, 0, 0, 0]])]
# )

# ids = [101, 1188, 1110, 102, 0, 0, 0]
# mask = [1, 1, 1, 1, 0, 0, 0]

def test_forward():
    n_labels = 7
    batch = 1
    expected_shape = (batch, n_labels)

    ids = [101, 1188, 1110, 102, 0, 0, 0]
    mask = [1, 1, 1, 1, 0, 0, 0]

    # id_list = np.random.randint(0,2500,768)
    # mask_list = np.random.randint(0,2,768)
    
    ids = torch.tensor(ids, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    # ids = ids.reshape(-1,1)
    # model = train_model.ModelTrainer(Distil_bert,0.01,5)
    # configuration = DistilBertConfig()
    # model = Distil_bert.classifier(ids)
    # model = nn.DataParallel(model)
    model = torch.load("models/model_epoch2.pth")
    # distil = DistilBertForSequenceClassification.from_pretrained(model_weights)
    # model = distil()
    model.eval()
    # print(model.shape)
    output = model(ids)
    # model.load_state_dict(torch.load("models/model_epoch2.pth",mask))
    # model = model_epoch2

    # actual = model.classifier()
    

    assert expected_shape == output.logits.shape, "Different shape"