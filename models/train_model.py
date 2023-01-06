from torch.optim import Adam
from tqdm import tqdm
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
import torch

class ModelTrainer:
    def __init__(self, model, learning_rate, epochs):

        self.model = model
        self.optimizer = Adam(params=model.parameters(), lr=learning_rate)
        self.Loss = BCELoss()
        self.scheduler = StepLR(self.optimizer, step_size=212, gamma=0.1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs 

    def train(self, model,Train_DL,Val_DL):
        
        model.to(self.device)
        model.train()
        
        train_acc_epochs = []
        train_loss_epochs = []
        val_acc_epochs = []
        val_loss_epochs = []
        
        for epoch in range(self.epochs):

            training_loss = {}
            training_accuracy = {}
            validation_loss = {}
            validation_accuracy = {}
            batch = 0
        
        for comments, labels in tqdm(Train_DL):
            
            labels = labels.to(self.device)
            labels = labels.float()
            masks = comments["attention_mask"].squeeze(1).to(self.device) # the model used these masks to attend only to the non-padded tokens in the sequence
            input_ids = comments["input_ids"].squeeze(1).to(self.device) # contains the tokenized and indexed representation for a batch of comments
            # squeeze is used to remove the second dimension which has size 1.
            output = model(input_ids, masks) # vector of logits for each class
            loss = self.Loss(output.logits, labels) # compute the loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            
            batch += 1 
            if batch%53 == 0:
                with torch.no_grad():
                    acc = []
                    op = output.logits
                    for lb in range(len(labels)): # note: labels is of shape (batch_size, num_classes(=7))
                        correct = 0
                        for i in range(len(labels[lb])):  # therefore len(labels[lb]) is 7
                            res = 1 if op[lb,i]>0.5 else 0
                            if res == labels[lb,i]:
                                correct += 1
                        acc.append(correct/len(labels[lb]))

                    training_loss[batch] = loss.item()
                    training_accuracy[batch] = sum(acc)/len(acc)
                    print(f"Epoch:{epoch+1} | batch no:{batch}/{len(Train_DL)} | Loss:{loss.item():.4f} | Accuracy:{sum(acc)/len(acc):.4f}")
        
                    # Testing model on validation Data
                    accVal = []
                    val_loss = 0
                    for comments, labels in Val_DL:
                        labels = labels.to(self.device)
                        labels = labels.float()
                        masks = comments["attention_mask"].squeeze(1).to(self.device)
                        input_ids = comments["input_ids"].squeeze(1).to(self.device)
    
                        output = model(input_ids, masks)
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
    
        return train_acc_epochs, train_loss_epochs, val_acc_epochs, val_loss_epochs

    
def Evaluate_Model(self, model,Test_DL):
    

    model.eval()
    
    accTest = []
    Test_loss = 0
    for comments, labels in Test_DL:
        labels = labels.to(self.device)
        labels = labels.float()
        masks = comments["attention_mask"].squeeze(1).to(self.device)
        input_ids = comments["input_ids"].squeeze(1).to(self.device)
    
        output = model(input_ids, masks)
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
    ModelTrainer()

