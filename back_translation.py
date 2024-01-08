import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import torch
from tqdm import tqdm

def load_config():
    with open('back_translation_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data():
    data_path = 'data/train_data.txt'
    label_path = 'data/train_label.txt'
    emo_data = [] # 保存需要增强的数据
    emo_label = [] # 保存对应的标签
    no_emo_data = [] # 保存不需要增强的数据
    no_emo_label = [] # 保存对应的标签
    with open(data_path, 'r') as f:
        lines = f.readlines()
        data = [line.strip() for line in lines] # 读所有数据
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            line = line.strip()
            if(line == '0,0,0,0,0,0'):#标签是000000就跳过
                no_emo_data.append(data[i])
                no_emo_label.append(line)
            else:#否则是增强目标
                emo_data.append(data[i])
                emo_label.append(line)
    return emo_data, emo_label, no_emo_data, no_emo_label

def translate(model, tokenizer, text, device) -> list[str]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def back_translation():
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    config = load_config()
    models = {}
    print('Loading NMT models...')
    for k in tqdm(config['model_paths'].keys()):
        path = config['model_paths'][k]
        models[k] = {
            'name': k,
            'path': path,
            'tokenizer': AutoTokenizer.from_pretrained(path) if os.path.exists(path) else None,
            'model': AutoModelForSeq2SeqLM.from_pretrained(path).to(device) if os.path.exists(path) else None,
        }
    tracks = [config['translation_tracks'][i] for i in config['active_tracks']]
    fold = len(tracks)+1
    emo_data, emo_label, no_emo_data, no_emo_label = load_data()
    # debug
    # emo_data = emo_data[:160-7]
    # emo_label = emo_label[:160-7]
    # no_emo_data = no_emo_data[:100]
    # no_emo_label = no_emo_label[:100]
    # debug end
    aug_data, aug_label = [], []
    batch_size = 16 # cuda memory usage: 26G
    print('Start translating...')
    # new_aug_data_temp = []
    # new_aug_label_temp = []
    for i in tqdm(range(0, len(emo_data), batch_size)):
        # all comments assume total data = 16000, batch_size = 32, context = 5, translation tracks = 3
        batch_samples = emo_data[i:i+batch_size] # 32
        batch_labels = emo_label[i:i+batch_size] # 32
        batch_texts_script = [] # 32*11 (1-dim)
        batch_texts_character = [] # 32*11 (the same every 11) (1-dim)
        aug_batch_samples = [] # gather all back translation results for this batch of samples
        for sample in batch_samples:
            texts = sample.split(', ') # 11
            texts_script = [text.split('角色：')[0] for text in texts] # 11
            texts_character = texts[0].split('角色：')[-1] # 1, the same for all 11
            batch_texts_script.extend(texts_script) # 32*11
            batch_texts_character.extend([texts_character] * len(texts_script)) # 32*11
        # translate
        zh_en = None # save frequently used translation result, can be extended to a dict
        for track in tracks:
            # e.g. track = [zh_en, en_fr, fr_en, en_zh]
            input = batch_texts_script
            for step in track:
                # e.g. step = en_fr
                if step == 'zh_en' and zh_en is not None:
                    output = zh_en
                    input = output
                    continue
                output = translate(models[step]['model'], models[step]['tokenizer'], input, device)
                input = output
                if zh_en is None and step == 'zh_en':
                    zh_en = output
            aug_batch_texts_script = output # 32*11 (1-dim)
            aug_batch_texts = [] # 32*11 (1-dim)
            for aug_text_script, texts_character in zip(aug_batch_texts_script, batch_texts_character):
                a_t_s = aug_text_script.strip('\'').strip('\"').strip('"').replace(', ', '，').replace(',', '，')
                t_c = texts_character.strip('\'').strip('\"').strip('"')
                aug_batch_texts.append("'" + a_t_s + '。角色：' + t_c + "'" )
            aug_batch_samples.extend(aug_batch_texts) # 3*32*11
        # join aug_batch_samples by (', ') every 11 as a group
        # so that aug_batch_samples is 3*32 (1-dim)
        # e.g. ['1', '2', ..., '11', '12', ... , '36']-> ['1, 2, ..., 11', '12, ..., 22', '23, ..., 33']
        aug_batch_samples = [', '.join(aug_batch_samples[i:i+11]) for i in range(0, len(aug_batch_samples), 11)] # 3*32
        aug_data.extend(batch_samples) # 1*32
        aug_data.extend(aug_batch_samples) # 3*32
        aug_label.extend(batch_labels) # 1*32
        aug_label.extend(batch_labels*(len(aug_batch_samples)//len(batch_labels))) # 3*32
        # emo_data.extend(aug_batch_samples) # 16000 + 3*32*(16000//32) = 16000*4 = 64000
        # emo_label.extend(batch_labels*(len(aug_batch_samples)//len(batch_labels))) # 64000
        # new_aug_data_temp.extend(aug_batch_samples)
        # new_aug_label_temp.extend(batch_labels*(len(aug_batch_samples)//len(batch_labels)))
        # if i > 2:
        #     break
    aug_label.extend(no_emo_label)
    aug_data.extend(no_emo_data)
    with open(f'data/train_data_bt_{fold}x.txt', 'w') as file:
        for item in aug_data:
        # for item in new_aug_data_temp:
            file.write(str(item)+ '\n')
    with open(f'data/train_label_bt_{fold}x.txt', 'w') as file:
        for item in aug_label:
        # for item in new_aug_label_temp:
            file.write(str(item)+ '\n')


if __name__ == '__main__':
    back_translation()
    pass