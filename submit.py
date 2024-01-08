from src.evaluate import do_evaluate

import torch

if __name__ == "__main__":
    weight_path=r"./weights/0.xxxx-dd-hhmm-epoch-batch.pt"
    model=torch.load(weight_path)
    do_evaluate(model)