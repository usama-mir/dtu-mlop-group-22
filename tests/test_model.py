# check if the dataset's first two columns are str, and rest are int type
from src.models import train_model

import torch
import torch.nn as nn
import pytest
import sys

sys.path.insert(0, "src")

# from model import MultilabelClassifier


@pytest.mark.parametrize(
    "ids, mask", [([[101, 1188, 1110, 102, 0, 0, 0]], [[1, 1, 1, 1, 0, 0, 0]])]
)
def test_forward(ids, mask):
    n_labels = 100
    batch = 1
    expected_shape = (batch, n_labels)

    model = train_model.ModelTrainer(n_labels)

    ids = torch.tensor(ids, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    actual = model.forward(ids=ids, mask=mask)

    assert expected_shape == actual.shape