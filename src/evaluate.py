import torch

import pandas as pd
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from collections import defaultdict
from networks import BERTAttentionsModel_v2
from transformers import  BertTokenizer
from datasets import TestDataset
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')


EPOCHS=5
sep=1
#权重位置
weight_path=r"./weights/0.7098-08-1300-3-2019.pt"
target_cols=['love', 'joy', 'fright', 'anger', 'fear', 'sorrow'] 
model_path=r"./chinese-roberta-wwm-ext"
batch_size=16
max_len = 128


def predict(model, test_loader):
    CUDA_DEVICE_ID = [4]
    cuda_master_id = CUDA_DEVICE_ID[0] if torch.cuda.is_available() and len(CUDA_DEVICE_ID) > 0 else None # first gpu, unless no gpu or forced to use cpu   
    device = torch.device("cuda:{}".format(cuda_master_id) if torch.cuda.is_available() else "cpu") # master (or the only) gpu, unless no gpu
    # device = torch.device("cpu")' # force to use cpu
    if device != torch.device("cpu") and len(CUDA_DEVICE_ID) > 1:
        model = nn.DataParallel(model, device_ids=CUDA_DEVICE_ID)
    model.to(device)
    model.eval()
    val_loss = 0
    test_pred = defaultdict(list)
    for batch in tqdm(test_loader):
        b_input_ids = batch['input_tokens'].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            logists = model(b_input_ids, attention_mask)
            for col in target_cols:
                out2 = logists[col].sigmoid().squeeze(1)*3.0
                test_pred[col].extend(out2.cpu().numpy().tolist())
    return test_pred


# #创建dataloader
# def create_dataloader(dataset, batch_size, mode='train'):
#     shuffle = True if mode == 'train' else False

#     if mode == 'train':
#         data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     else:
#         data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return data_loader


def do_evaluate(model):

    #提交模板
    submit = pd.read_csv('data/submit_example.tsv', sep='\t')

    #加载测试集
    tokenizer = BertTokenizer.from_pretrained(model_path)
    valset = TestDataset(tokenizer, max_len,sep, mode='test')
    # valid_loader = create_dataloader(valset, batch_size, mode='test')
    valid_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    #预测
    test_pred = predict(model, valid_loader)

    #变为指定格式
    index=pd.read_csv(f"data/test_processed_sep_{sep}_sorted.csv",sep='\t')['id']
    df1=pd.DataFrame(test_pred,index=index)
    arr=df1[target_cols].to_numpy()
    df1['emotion']=[",".join(map(str,r)) for r in arr]
    submit['emotion']=df1.loc[submit['id'].tolist(),'emotion'].tolist()
  
    #保存
    submit.to_csv(f'./results/baseline_BCE_epoch{EPOCHS}_sep{sep}_sorted_v2.tsv', sep='\t', index=False)
