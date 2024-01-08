# Attention-Bert-Contextual-Data-augmentation-Emotion-recognition (ABCDE)

Project for the course of Advanced Artificial Intelligence (AAI), University of Chinese Academy of Sciences (UCAS), 2023 Autumn

[DataFountain Competition](https://www.datafountain.cn/competitions/518)

1. run `get_train_data.py` for preprocessing the data
2. (optional) download NMT models from [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) for back translation and run `back_translation.py` to augment the data
3. download BERT base model from [
chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) and
run `main.py` to train the model
4. (optional) run `submit.py` to generate the submission file to DataFountain
