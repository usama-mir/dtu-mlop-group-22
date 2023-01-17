from fastapi import FastAPI, Request, Response
import torch 
import sys 
import os 
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


app = FastAPI()

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
   return {"item_id": item_id}


model = torch.load("../models/model_epoch2.pth")
model.eval()


# Defining the prediction endpoint without data validation
@app.get('/basic_predict')
async def basic_predict(comment: str):
    
    # Getting the JSON from the body of the request


    tk = tokenizer(comment, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = tk['input_ids']
    attention_masks = tk['attention_mask']

    ids = torch.tensor(input_ids, dtype=torch.long)
    mask = torch.tensor(attention_masks, dtype=torch.long)

    output = model(input_ids = ids.reshape(-1,1),attention_mask=mask)
    output = output.logits.detach()
    print('reached this line')
    #print(output.numpy())

    # Converting JSON to Pandas DataFrame
    #input_processed = pd.DataFrame([input_raw])
    
    # Getting the prediction from the Logistic Regression model
    #pred = lr_model.predict(input_processed)[0]
 
    return Response(content=str(output.numpy()))

