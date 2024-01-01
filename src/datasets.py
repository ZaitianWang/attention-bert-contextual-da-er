#可用的dataset和dataloader
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import BertPreTrainedModel
from transformers import BertTokenizer
import torch
import pandas as pd
import numpy as np

class ContextualDataset(Dataset):
    # def __init__(self, tokenizer, max_len, mode='train'):
    def __init__(self, data, label, tokenizer, max_len, mode='train'):
        super(ContextualDataset, self).__init__()
        if mode == 'train':
            # file = './train_data.txt'
            # with open(file,'r') as f:
            #     text_list = f.readlines()
            #     self.data = [line.strip() for line in text_list]
                
            # file = './train_label.txt'
            # with open(file,'r') as f:
            #     label_list = f.readlines()
            #     self.label = [line.strip() for line in label_list]
            self.data = data
            self.label = label
        else:
            self.data = pd.read_csv('data/test.csv',sep='\t')
            
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

        win_sen = self.data[index].split(', ')
        if(len(win_sen)!=7):
            # print("error")
            # only keep first 7 sentences
            win_sen = win_sen[:7]
        #each[1:-1]去前后引号，空白格
        for i,each in enumerate(win_sen):
            each = each[1:-1]
            # if(i==0):
            #     each = each[1:-1]
            # else:
            #     each = each[2:-1]
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
    
    
# PRE_TRAINED_MODEL_NAME='hfl/chinese-roberta-wwm-ext'  # 'hfl/chinese-roberta-wwm-ext'
# tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
# trainset = RoleDataset(tokenizer, max_len=128, mode='train')
# dataloader = DataLoader(trainset)