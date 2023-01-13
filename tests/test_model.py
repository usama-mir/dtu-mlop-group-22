from src.models.model import Distil_bert
from src.models import train_model
from transformers import DistilBertTokenizer, DistilBertForTokenClassification, DistilBertConfig
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
    batch = 32
    expected_shape = (batch, n_labels)

    ids = [101, 1188, 1110, 102, 0, 0, 0]
    mask = [1, 1, 1, 1, 0, 0, 0]

    id_list = np.random.randint(0,2500,768)
    mask_list = np.random.randint(0,2,768)

    ids = torch.tensor(id_list, dtype=torch.int)
    mask = torch.tensor(mask_list, dtype=torch.int)
    
    # model = train_model.ModelTrainer(Distil_bert,0.01,5)
    configuration = DistilBertConfig(vocab_size=(1,768))
    model = Distil_bert(configuration)
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load("models/model_epoch2.pth"))
    # model.eval()
    output = model.forward(ids,mask)
    # model.load_state_dict(torch.load("models/model_epoch2.pth",mask))
    # model = model_epoch2

    # actual = model.classifier()
    

    assert expected_shape == output.shape, "Different shape"