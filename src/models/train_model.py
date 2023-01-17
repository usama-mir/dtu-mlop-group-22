import sys
import os
import wandb

sys.path.append("./src/data")
sys.path.append("../../data")

import hydra
from hydra.utils import get_original_cwd
from torch.optim import Adam
from tqdm import tqdm
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
import torch
from model import Distil_bert
import pandas as pd
from dataset import Toxic_Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Dict


class ModelTrainer:
    def __init__(self, cfg, model: Distil_bert) -> None:
        """
        Initialize the model trainer

        :param model: The model to be trained
        :param learning_rate: The learning rate for the optimizer
        :param epochs: The number of epochs for training
        """
        self.model = model
        self.optimizer = Adam(params=model.parameters(), lr=cfg.hyperparameters.lr)
        self.Loss = BCELoss()
        self.scheduler = StepLR(self.optimizer, step_size=cfg.hyperparameters.step_size, gamma=cfg.hyperparameters.gamma)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = cfg.hyperparameters.epochs  
        self.n_classes = cfg.hyperparameters.n_classes

    def train(self, Train_DL: DataLoader, Val_DL: DataLoader) -> None:
        """
        Train the model on the input data

        :param Train_DL: DataLoader for the training dataset
        :param Val_DL: DataLoader for the validation dataset
        """
        self.model.to(self.device)
        self.model.train()

        train_acc_epochs = []
        train_loss_epochs = []
        val_acc_epochs = []
        val_loss_epochs = []

        for epoch in range(self.epochs):
            print(epoch)
            training_loss = {}
            training_accuracy = {}
            validation_loss = {}
            validation_accuracy = {}
            batch = 0

            for comments, labels in tqdm(Train_DL):
                labels = labels.float().to(self.device)
                masks = (
                    comments["attention_mask"].squeeze(1).to(self.device)
                )  # the model used these masks to attend only to the non-padded tokens in the sequence
                input_ids = (
                    comments["input_ids"].squeeze(1).to(self.device)
                )  # contains the tokenized and indexed representation for a batch of comments
                output = self.model(input_ids, masks)  # vector of logits for each class
                loss = self.Loss(output.logits, labels)  # compute the loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                print(f" Train Loss:{loss/len(Train_DL):.4f}")
                wandb.log({"Training loss": loss/len(Train_DL)})

                with torch.no_grad():
                    # Testing model on validation
                    accVal = []
                    val_loss = 0
                    for comments, labels in Val_DL:
                        labels = labels.float().to(self.device)
                        masks = comments["attention_mask"].squeeze(1).to(self.device)
                        input_ids = comments["input_ids"].squeeze(1).to(self.device)

                        output = self.model(input_ids, masks)
                        loss = self.Loss(output.logits, labels)
                        val_loss += loss.item()

                        op = output.logits
                        correct_val = 0
                        for i in range(self.n_classes):
                            res = 1 if op[0, i] > 0.5 else 0
                            if res == labels[0, i]:
                                correct_val += 1
                        accVal.append(correct_val / self.n_classes)

                    validation_loss[batch] = val_loss / len(Val_DL)
                    validation_accuracy[batch] = sum(accVal) / len(accVal)
                    print(
                        f" Validation Loss:{val_loss/len(Val_DL):.4f} | Validation Accuracy:{sum(accVal)/len(accVal):.4f}"
                    )
                    wandb.log({"Validation loss": loss/len(Train_DL)})

                train_acc_epochs.append(training_accuracy)
                train_loss_epochs.append(training_loss)
                val_acc_epochs.append(validation_accuracy)
                val_loss_epochs.append(validation_loss)

        torch.save(
            self.model.state_dict(), "models/model_epoch{}.pth".format(self.epochs)
        )

        return train_acc_epochs, train_loss_epochs, val_acc_epochs, val_loss_epochs

    def evaluate_model(self, Test_Dl: DataLoader):
        """
        Evaluate the model on the test data

        :param Train_DL: DataLoader for the evaluating dataset
        """
        self.model.eval()

        accTest = []
        Test_loss = 0
        for comments, labels in Test_Dl:
            labels = torch.from_numpy(labels).to(self.device)
            labels = labels.float()
            masks = comments["attention_mask"].squeeze(1).to(self.device)
            input_ids = comments["input_ids"].squeeze(1).to(self.device)

            output = self.model(input_ids, masks)
            loss = self.Loss(output.logits, labels)
            Test_loss += loss.item()

            op = output.logits
            correct_val = 0
            for i in range(self.n_classes):
                res = 1 if op[0, i] > 0.5 else 0
                if res == labels[0, i]:
                    correct_val += 1
            accTest.append(correct_val / self.n_classes)

        print("Testing Dataset:\n")
        print(
            f" Test Loss:{Test_loss/len(Test_Dl):.4f} | Test Accuracy:{sum(accTest)/len(accTest):.4f}"
        )


@hydra.main(config_name="config.yaml")
def main(cfg:Dict) -> None:
    wandb.init(project="test-project", entity="dtu_mloperations")
    wandb.config = cfg

    trainer = ModelTrainer(cfg, Distil_bert)
    # now run the data through the toxic dataset
    # then call the train function and hope for the best
    original_cwd = get_original_cwd()
    data = pd.read_csv(os.path.join(original_cwd, "data/processed/train_processed.csv"), nrows=1000)

    X_train, X_val, Y_train, Y_val = train_test_split(
        pd.DataFrame(data.iloc[:, 1]),
        pd.DataFrame(data.iloc[:, 2:]),
        test_size=0.1,
        stratify=data.iloc[:, 9],
    )
    Y_train.drop(columns=["total_classes"], inplace=True)
    Y_val.drop(columns=["total_classes"], inplace=True)

    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_val = pd.DataFrame(X_val).reset_index(drop=True)
    Y_train = pd.DataFrame(Y_train).reset_index(drop=True)
    Y_val = pd.DataFrame(Y_val).reset_index(drop=True)

    # drop total classes
    # Making Training, Testing and Validation of data using Dataset class
    Train_data = Toxic_Dataset(X_train, Y_train)
    Val_data = Toxic_Dataset(X_val, Y_val)

    Train_DL = Toxic_Dataset(X_train, Y_train)
    Train_Loader = DataLoader(Train_DL, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    Val_DL = Toxic_Dataset(X_val, Y_val)
    Val_Loader = DataLoader(Val_DL, batch_size=cfg.hyperparameters.batch_size, shuffle=True)

    trainer.train(Train_Loader, Val_Loader)


if __name__ == "__main__":
    main()
