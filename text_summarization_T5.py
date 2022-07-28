# Install the hugging face transformer and pytorch lighting
#!pip install  transformers
#!pip install  pytorch-lightning

#importing the library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader,Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from termcolor import colored
import textwrap
from transformers import AdamW,T5ForConditionalGeneration,T5TokenizerFast
import tqdm.auto as tqdm
from pylab import rcParams
#set the seed value
pl.seed_everything(42)

#load the dataset 
data=pd.read_csv("news_summary.csv",encoding="latin-1")

#grab only the complete and summarize text columns
df=data[["text","ctext"]]

#rename the columns
df.columns=["summary","text"]
df.dropna()
print(df.shape)

#train test split the data
train_df,test_df=train_test_split(df,test_size=0.1)
print(train_df.shape,test_df.shape)

#initialize the tokenizer
model_name="t5-base"
tokenizer=T5TokenizerFast.from_pretrained(model_name)

# create the model inputs (tokenized the data)
class NewsSummaryDataset(Dataset):
    def __init__(self,
                data:pd.DataFrame,
                tokenizer: T5TokenizerFast,
                text_max_token_len: int=512,
                summary_max_token_len: int=128):
        
        self.tokenizer=tokenizer
        self.data=data
        self.text_max_token_len=text_max_token_len
        self.summary_max_token_len=summary_max_token_len
        
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self,index: int):
        data_row=self.data.iloc[index]
        
        text=data_row["text"]
        summary=data_row["summary"]
        
        text_encoding=tokenizer(
        text,
        max_length=self.text_max_token_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt")
        
        summary_encoding=tokenizer(
        summary,
        max_length=self.summary_max_token_len, #create the encodings vector of fixed length
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt")
        
        labels=summary_encoding["input_ids"] # input_ids=unique id of each token
        labels[labels==0]=-100 # avoid the computations of padding loss.  
        
        return dict(
        text=text,
        summary=summary,
        text_input_ids=text_encoding["input_ids"].flatten(),
        text_attention_mask=text_encoding["attention_mask"].flatten(),
        labels=labels.flatten(),
        labels_attention_mask=summary_encoding["attention_mask"].flatten())

# create the train + test tokenized dataset and train and test dataloader
class NewsSummaryDataModule(pl.LightningDataModule):
    def __init__(self,
                train_df:pd.DataFrame,
                test_df:pd.DataFrame,
                tokenizer:T5TokenizerFast,
                batch_size: int=8,
                text_max_token_len: int=512,
                summary_max_token_len: int=128):
        super().__init__()
        
        self.train_df=train_df
        self.test_df=test_df
        self.batch_size=batch_size
        self.tokenizer=tokenizer
        self.text_max_token_len=text_max_token_len
        self.summary_max_token_len=summary_max_token_len
        
        #create dataset
    def setup(self,stage=None):
        self.train_dataset=NewsSummaryDataset(
        self.train_df,
        self.tokenizer,
        self.text_max_token_len,
        self.summary_max_token_len)
        
        self.test_dataset=NewsSummaryDataset(
        self.test_df,
        self.tokenizer,
        self.text_max_token_len,
        self.summary_max_token_len)
            
    #create the dataloader
    def train_dataloader(self):
        return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True
        )
    def val_dataloader(self):
        return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        shuffle=False
        )


NB_EPOCHS=3
BATCH_SIZE=8

#initialize the data module .
data_module=NewsSummaryDataModule(train_df,test_df,tokenizer,batch_size=BATCH_SIZE)

#----------------Building the model-------------------------------------

class NewsSummaryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model=T5ForConditionalGeneration.from_pretrained(model_name,return_dict=True)
        
    def forward(self,input_ids,attention_mask,decoder_attention_mask,labels=None):
        output=self.model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels, # already calculated .
        decoder_attention_mask=decoder_attention_mask    
        )
        
        return output.loss,output.logits
    
     # To complete training loop(batch wise)
    def training_step(self,batch,batch_idx): 
        input_ids=batch["text_input_ids"]
        attention_mask=batch["text_attention_mask"]
        labels=batch["labels"]
        labels_attention_mask=batch["labels_attention_mask"]
        
        # call the forward function
        loss,outputs=self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels=labels)
        
       
        self.log("train_loss",loss,prog_bar=True,logger=True)
        return loss
    
    # To complete validation loop(batch wise)
    def validation_step(self,batch,batch_idx):
        input_ids=batch["text_input_ids"]
        attention_mask=batch["text_attention_mask"]
        labels=batch["labels"]
        labels_attention_mask=batch["labels_attention_mask"]
        
        # call the forward function
        loss,outputs=self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels=labels)
        
       
        self.log("validation_loss",loss,prog_bar=True,logger=True)
        return loss
    
    # To complete test loop(batch wise)
    def test_step(self,batch,batch_idx): 
        input_ids=batch["text_input_ids"]
        attention_mask=batch["text_attention_mask"]
        labels=batch["labels"]
        labels_attention_mask=batch["labels_attention_mask"]
        
        # call the forward function
        loss,outputs=self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels=labels)
        
       
        self.log("test_loss",loss,prog_bar=True,logger=True)
        return loss
    
    # define optimizers and LR schedulers
    def configure_optimizers(self):
        return AdamW(self.parameters(),lr=0.0001)
    
# Initialize the T5 model .   
model=NewsSummaryModel()

#%load_ext tensorboard
#%tensorboard --logdir ./lightning_logs

# build the model chaeckpoints and logger
check_point_callback=ModelCheckpoint(
                        dirpath="checkpoints",
                        filename="best_checkpoint",
                        save_top_k=1,
                        verbose=True,
                        monitor="val_loss",
                        mode="min")

logger= TensorBoardLogger("lightning_logs",name="news-summary")

#Trainer handles the training loop details
trainer=pl.Trainer(logger=logger,
                   checkpoint_callback=check_point_callback,
                  max_epochs=NB_EPOCHS)

# start the training process
trainer.fit(model,data_module)

#load our trained model
trained_model=NewsSummaryModel.load_from_checkpoint(
                       trainer.checkpoint_callback.best_model_path)

trained_model.freeeze()

#Create the summary of the given text input.
def summarize(text):
    text_encoding=tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt")
    
    # Use of the trained model for generate the text summerization.
    generated_ids=trained_model.model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=100,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True ) 
    
    preds = [
        tokenizer.decode(gen_id,skip_special_tokens=True,clean_up_tokenization_spaces=True) for gen_id in generated_ids
    ]
    
    return "".join(preds)

#------------------- show the predictions---------------------------------

sample_row=test_df.iloc[0]
text=sample_row["text"]
model_output=summarize(text)
print("text :",text)
print("Original_summary:",sample_row["summary"])  
print("Predicted_summary:",model_output) 
