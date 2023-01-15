# # test which checks that data gets correctly loaded
# from src.data import make_dataset
import sys
sys.path.append("./src/data")
import pandas as pd 
import torch 
from dataset import Toxic_Dataset
def test_data():

    n_rows = 31513
    dataset = pd.read_csv("./data/processed/train_processed.csv",nrows=100)
    #assert len(dataset) == n_rows

    attributes = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate', 'total_classes', 'non_toxic']
    assert attributes == list(dataset.columns)

    Train_data = Toxic_Dataset(pd.DataFrame(dataset[['comment_text']]), pd.DataFrame(dataset[['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate', 'total_classes', 'non_toxic']]))


    comment, label = Train_data.__getitem__(10)
    comment = torch.tensor(comment, dtype=torch.long)
    label = torch.tensor(label, dtype=torch.long)
    print(type(comment.size()))
    print(type(label.size()))

if __name__ == '__main__':
    test_data()
