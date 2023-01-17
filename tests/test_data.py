# # test which checks that data gets correctly loaded
# from src.data import make_dataset
import sys

sys.path.append("./src/data")
import pandas as pd
import torch
from dataset import Toxic_Dataset


def test_data():

    n_rows = 31513
    dataset = pd.read_csv("./data/processed/train_processed.csv")
    assert len(dataset) == n_rows, "size of rows are different"

    attributes = [
        "id",
        "comment_text",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
        "total_classes",
        "non_toxic",
    ]
    assert attributes == list(dataset.columns), "cols are different"

    Train_data = Toxic_Dataset(
        pd.DataFrame(dataset[["comment_text"]]),
        pd.DataFrame(
            dataset[
                [
                    "toxic",
                    "severe_toxic",
                    "obscene",
                    "threat",
                    "insult",
                    "identity_hate",
                    "total_classes",
                    "non_toxic",
                ]
            ]
        ),
    )
    comment, label = Train_data.__getitem__(15)
    comment = torch.tensor(comment["input_ids"].clone().detach(), dtype=torch.long)
    label = torch.tensor(label, dtype=torch.long)
    assert comment.size() == torch.Size([1, 512]), "comment size is different"
    assert label.size() == torch.Size([8]), "label size is different"


if __name__ == "__main__":
    test_data()
