from src.evaluate import do_evaluate

import torch

if __name__ == "__main__":
    weight_path=r"./weights/0.xxxx-dd-hhmm-epoch-batch.pt"
    weight_path=r"./weights/0.6955-07-1956-3-2895.pt"
    model=torch.load(weight_path)
    do_evaluate(model)