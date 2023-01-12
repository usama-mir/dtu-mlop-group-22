import sys 
sys.path.append('./src/data')

from torch.optim import Adam
from tqdm import tqdm
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
import torch
from model import Distil_bert
import pandas as pd 
from dataset import Toxic_Dataset
from sklearn.model_selection import train_test_split
import numpy as np 
from torch.utils.data import DataLoader, Dataset

class ModelTrainer:
    def __init__(self, model, learning_rate, epochs):

        self.model = model
        self.optimizer = Adam(params=model.parameters(), lr=learning_rate)
        self.Loss = BCELoss()
        self.scheduler = StepLR(self.optimizer, step_size=212, gamma=0.1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 2 #epochs 

    def train(self,Train_DL,Val_DL):
        
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
                masks = comments["attention_mask"].squeeze(1).to(self.device) # the model used these masks to attend only to the non-padded tokens in the sequence
                input_ids = comments["input_ids"].squeeze(1).to(self.device) # contains the tokenized and indexed representation for a batch of comments
                print('input_ids SHAPE {}'.format(input_ids.shape))
                output = self.model(input_ids, masks) # vector of logits for each class
                loss = self.Loss(output.logits, labels) # compute the loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                print(f" Train Loss:{loss/len(Train_DL):.4f}")
                
                with torch.no_grad():
                    # Testing model on validation Data
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
                        for i in range(7):
                            res = 1 if op[0,i]>0.5 else 0
                            if res == labels[0,i]:
                                correct_val += 1
                        accVal.append(correct_val/7)
                    
                    validation_loss[batch] = val_loss/len(Val_DL)
                    validation_accuracy[batch] = sum(accVal)/len(accVal)
                    print(f" Validation Loss:{val_loss/len(Val_DL):.4f} | Validation Accuracy:{sum(accVal)/len(accVal):.4f}")
            
                train_acc_epochs.append(training_accuracy)
                train_loss_epochs.append(training_loss)
                val_acc_epochs.append(validation_accuracy)
                val_loss_epochs.append(validation_loss)

        torch.save(self.model.state_dict(), "models/model_epoch{}.pth".format(self.epochs))
    
        return train_acc_epochs, train_loss_epochs, val_acc_epochs, val_loss_epochs

    
    def Evaluate_Model(self,Test_DL):
        

        self.model.eval()
        
        accTest = []
        Test_loss = 0
        for comments, labels in Test_DL:
            labels = torch.from_numpy(labels).to(self.device)
            labels = labels.float()
            masks = comments["attention_mask"].squeeze(1).to(self.device)
            input_ids = comments["input_ids"].squeeze(1).to(self.device)
        
            output = self.model(input_ids, masks)
            loss = self.Loss(output.logits, labels)
            Test_loss += loss.item()
                
            op = output.logits
            correct_val = 0
            for i in range(7):
                res = 1 if op[0,i]>0.5 else 0
                if res == labels[0,i]:
                    correct_val += 1
            accTest.append(correct_val/7)
        
        print("Testing Dataset:\n")
        print(f" Test Loss:{Test_loss/len(Test_DL):.4f} | Test Accuracy:{sum(accTest)/len(accTest):.4f}")


if __name__ == "__main__":
    trainer = ModelTrainer(Distil_bert, 0.01, 10)
    import os 
    #print(os.getcwd())
    
    
    #now run the data through the toxic dataset 
    #then call the train function and hope for the best
    data = pd.read_csv('./data/processed/train_processed.csv', nrows=1000)
    #print(data.head(10))
    #print(data.head(20))
    
    X_train, X_val, Y_train, Y_val = train_test_split(pd.DataFrame(data.iloc[:,1]),pd.DataFrame(data.iloc[:,2:]), test_size=0.1, stratify=data.iloc[:,9])
    Y_train.drop(columns=["total_classes"], inplace=True)
    Y_val.drop(columns=["total_classes"], inplace=True)
    # print(Y_train.columns)
    # print(Y_val.columns)
    # print()
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_val = pd.DataFrame(X_val).reset_index(drop=True)
    Y_train = pd.DataFrame(Y_train).reset_index(drop=True)
    Y_val = pd.DataFrame(Y_val).reset_index(drop=True)
    X_train.to_csv('X_train.csv')

    #drop total classes 
    # Making Training, Testing and Validation of data using Dataset class
    Train_data = Toxic_Dataset(X_train, Y_train)
    #Test_data = Toxic_Dataset(X_test, Y_test)
    Val_data = Toxic_Dataset(X_val, Y_val)

    Train_DL  = Toxic_Dataset(X_train, Y_train)
    Train_Loader = DataLoader(Train_DL, batch_size=32, shuffle=True)
    Val_DL = Toxic_Dataset(X_val, Y_val)
    Val_Loader = DataLoader(Val_DL, batch_size=32, shuffle=True)
    #print(X_train)
    #print(X_train.to_frame().columns)
    
    trainer.train(Train_Loader,Val_Loader)
