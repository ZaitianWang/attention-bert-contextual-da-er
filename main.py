from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup

from src.metrics import Criterion
from src.datasets import ContextualDataset
from src.networks import IQIYModelLite, BERTAttentionsModel_v2
from src.utils import plot_performance
from src.train import do_train
from src.evaluate import do_evaluate

'''
@Author: Wang
'''

CUDA_DEVICE_ID = [4]
cuda_master_id = CUDA_DEVICE_ID[0] if torch.cuda.is_available() and len(CUDA_DEVICE_ID) > 0 else None # first gpu, unless no gpu or forced to use cpu   
device = torch.device("cuda:{}".format(cuda_master_id) if torch.cuda.is_available() else "cpu") # master (or the only) gpu, unless no gpu
# device = torch.device("cpu")' # force to use cpu

#超参数
EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-5
WEIGHT_DACAY = 0.0
WARMUP_PROPORTION = 0.0
WARM_UP_RATIO = 0
LOSS_FUNC = 'RMSE' # 损失函数RMSE
LOG_STEP = 100 # aka validation step（每100步验证一次）
LEAST_ACCEPT_SCORE = 0.66 # 最低接受分数
TOY = False

SUBMIT = False

train_option = {
    'epochs': EPOCHS,
    'LOG_STEP': LOG_STEP,
    'LEAST_ACCEPT_SCORE': LEAST_ACCEPT_SCORE,
    'TOY': TOY
}


# DATA_FILE_PATH = 'data/train_data.txt'
# LABEL_FILE_PATH = 'data/train_label.txt'
DATA_FILE_PATH = 'data/train_data_bt_2x.txt'
LABEL_FILE_PATH = 'data/train_label_bt_2x.txt'
MAX_LEN = 128
CONTEXT_LEN = 1 # 0-5可选

#读数据
with open(DATA_FILE_PATH,'r') as f:
    text_list = f.readlines()
    data = [line.strip() for line in text_list]
with open(LABEL_FILE_PATH,'r') as f:
    label_list = f.readlines()
    label = [line.strip() for line in label_list]

#随机划分，自制 验证集10%
train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=0.1, random_state=42)
#预训练模型
PRE_TRAINED_MODEL_PATH='./chinese-roberta-wwm-ext/'  # 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_PATH)
base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_PATH)  # 加载预训练模型

train_set = ContextualDataset(train_data, train_label, tokenizer, MAX_LEN, CONTEXT_LEN)
valid_set = ContextualDataset(valid_data, valid_label, tokenizer, MAX_LEN, CONTEXT_LEN)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)


#使用basemodel
baseline = IQIYModelLite(n_classes=1, model_name=PRE_TRAINED_MODEL_PATH)
# if device != torch.device("cpu") and len(CUDA_DEVICE_ID) > 1:
#     baseline = nn.DataParallel(baseline, device_ids=CUDA_DEVICE_ID)
# baseline.to(device)
# model = baseline

proposed = BERTAttentionsModel_v2(n_classes=1, model_name=PRE_TRAINED_MODEL_PATH)
if device != torch.device("cpu") and len(CUDA_DEVICE_ID) > 1:
    proposed = nn.DataParallel(proposed, device_ids=CUDA_DEVICE_ID)
proposed.to(device)
model = proposed 

criterion = Criterion(loss_fn=LOSS_FUNC)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DACAY)
total_steps = len(train_loader) * EPOCHS
warm_up_steps = int(total_steps * WARM_UP_RATIO)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps, num_training_steps=total_steps)

train_checkpoint_performance, valid_checkpoint_performance = do_train(model, train_loader, valid_loader, optimizer, scheduler, criterion, device, train_option)

plot_performance(train_checkpoint_performance, valid_checkpoint_performance)

if SUBMIT:
    do_evaluate(model)
