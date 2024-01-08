#可用的dataset和dataloader
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import BertPreTrainedModel
from transformers import BertTokenizer
import torch
import pandas as pd
import numpy as np

'''
@Author: Huang, Zhang
'''

class ContextualDataset(Dataset):
    # def __init__(self, tokenizer, max_len, mode='train'):
    def __init__(self, data, label, tokenizer, max_len,context_len=5):
        super(ContextualDataset, self).__init__()
        self.data = data
        self.label = label
        self.context_len = context_len
            
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        target_cols=['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']
        input_ids = []
        attention_mask = []
        sample = {}
        labels = self.label[index].split(',')
        for k,label_col in enumerate(target_cols):
            sample[label_col] = torch.tensor(int(labels[k])/3.0,dtype=torch.float)

        sentences = self.data[index].split(', ')
        win_sen = [sentences[0]]
        con_len = self.context_len

        start = int(len(sentences) / 2) - con_len
        # input_seq = [0]
        for i in range(con_len*2):
            win_sen.append(sentences[start+i+1])
        # print(input_seq)

        for each in win_sen:
            each = each[1:-1]
            encoding = self.tokenizer.encode_plus(each,
                                            add_special_tokens=True, # 加上开头结尾符
                                            truncation=True,
                                            max_length = self.max_len, # 设置最大长度，与pad合用
                                            padding = 'max_length',
                                            return_token_type_ids=True,
                                            return_attention_mask=True,
                                            return_tensors='pt')
            input_ids.append(encoding['input_ids'].flatten().numpy())
            attention_mask.append(encoding['attention_mask'].flatten().numpy())
            
        sample['input_ids'] = torch.tensor(np.array(input_ids))
        sample['attention_mask'] = torch.tensor(np.array(attention_mask))
        return sample

    def __len__(self):
        return len(self.data)
    


class TestDataset(Dataset):
    def __init__(self, tokenizer, max_len,sep, mode='train'):
        super(TestDataset, self).__init__()
        #定义dataset
        self.target_cols=['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']
        with open(r"./data/test_data.txt",'r') as f:
            text_list = f.readlines()
        self.data = [eval(f"[{line.strip()}]") for line in text_list]
        self.tokenizer = tokenizer
        self.max_len = max_len
    

    def __getitem__(self, index):
        samples=self.data[index]
        encodings=[self.tokenizer.encode_plus(s,
                                            add_special_tokens=True,
                                            truncation=True,
                                            max_length=self.max_len,
                                            padding = 'max_length',
                                            return_token_type_ids=True,
                                            return_attention_mask=True,
                                            return_tensors='pt',) for s in samples]
        
        texts=[s for s in samples]
        input_tokens=np.array([encoding['input_ids'].flatten() for encoding in encodings],dtype=np.longlong)
        attention_masks=np.array([encoding['attention_mask'].flatten()  for encoding in encodings],dtype=np.longlong)
        sample = {
            'texts': texts, #原始文字
            'input_tokens': torch.tensor(input_tokens), #输入tokens
            'attention_mask': torch.tensor(attention_masks,dtype=torch.long)
        }
        
        return sample

    def __len__(self):
        return len(self.data)


