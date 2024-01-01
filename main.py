from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup

from src.metrics import Criterion
from src.datasets import ContextualDataset
from src.networks import IQIYModelLite, BERTAttentionsModel_v1
from src.utils import plot_performance
from src.train import do_train


CUDA_DEVICE_ID = [4]
cuda_master_id = CUDA_DEVICE_ID[0] if torch.cuda.is_available() and len(CUDA_DEVICE_ID) > 0 else None # first gpu, unless no gpu or forced to use cpu   
device = torch.device("cuda:{}".format(cuda_master_id) if torch.cuda.is_available() else "cpu") # master (or the only) gpu, unless no gpu
# device = torch.device("cpu")' # force to use cpu

EPOCHS = 1
BATCH_SIZE = 8
LR = 1e-5
WEIGHT_DACAY = 0.0
WARMUP_PROPORTION = 0.0
WARM_UP_RATIO = 0
LOSS_FUNC = 'BCE'
LOG_STEP = 100 # aka validation step
LEAST_ACCEPT_SCORE = 0.65
TOY = True 

train_option = {
    'epochs': EPOCHS,
    'LOG_STEP': LOG_STEP,
    'LEAST_ACCEPT_SCORE': LEAST_ACCEPT_SCORE,
    'TOY': TOY
}


DATA_FILE_PATH = 'data/train_data.txt'
LABEL_FILE_PATH = 'data/train_label.txt'
MAX_LEN = 128

with open(DATA_FILE_PATH,'r') as f:
    text_list = f.readlines()
    data = [line.strip() for line in text_list]
with open(LABEL_FILE_PATH,'r') as f:
    label_list = f.readlines()
    label = [line.strip() for line in label_list]

train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=0.1, random_state=42)

PRE_TRAINED_MODEL_PATH='./chinese-roberta-wwm-ext/'  # 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_PATH)
base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_PATH)  # 加载预训练模型

train_set = ContextualDataset(train_data, train_label, tokenizer, MAX_LEN, mode='train')
valid_set = ContextualDataset(valid_data, valid_label, tokenizer, MAX_LEN, mode='train')
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)

# # consistent with both single gpu and multi-gpu
baseline = IQIYModelLite(n_classes=1, model_name=PRE_TRAINED_MODEL_PATH)
# if device != torch.device("cpu") and len(CUDA_DEVICE) > 1:
#     baseline = nn.DataParallel(baseline, device_ids=CUDA_DEVICE)
# baseline.to(device)
# model = baseline
proposed = BERTAttentionsModel_v1(n_classes=1, model_name=PRE_TRAINED_MODEL_PATH)
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