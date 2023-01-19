from transformers import DistilBertTokenizer
import torch
import sys

sys.path.insert(0, "src")


def test_forward():

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    comments = "Bye! Don't look, come or think of comming back! Tosser."

    tk = tokenizer(comments, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = tk["input_ids"]
    attention_masks = tk["attention_mask"]

    len_id = len(input_ids[0])
    labels = 7
    expected_shape = (len_id, labels)
    print(expected_shape)

    ids = torch.tensor(input_ids, dtype=torch.long)
    mask = torch.tensor(attention_masks, dtype=torch.long)

    model = torch.load("models/model_epoch2.pth")

    model.eval()

    output = model(input_ids=ids.reshape(-1, 1), attention_mask=mask)

    print(output.logits.shape)
    assert expected_shape == output.logits.shape, "Different shape"