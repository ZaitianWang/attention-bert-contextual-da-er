import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

'''
@Author: Wang
'''

class Criterion():
    def __init__(self, loss_fn='BCE'):
        self.loss_fn = loss_fn
    def __call__(self, outputs, labels):
        loss_fn = self.loss_fn
        BCE = nn.BCEWithLogitsLoss()
        MSE = nn.MSELoss()
        # calculate bce, mse, rmse, score, acc, precision, recall, f1
        # the chosen loss function is returned as 'loss'
        # all are returned as a dict
        bce, mse = 0, 0
        clf_report = {} # returned as macro over emotions, dict for each emotion not returned
        for k in outputs.keys(): # k: for each emotion
            bce += BCE(outputs[k], labels[k].view(-1,1))
            # bce with logits already contains sigmoid, but mse doesn't
            pred = torch.sigmoid(outputs[k])*3.0
            ture = labels[k].view(-1,1)*3.0
            mse += MSE(pred, ture)
            # mse uses number in [0,3], but classification_report uses 0/1, threshold is 0.5
            pred_lbl = torch.where(pred>0.5, 1, 0)
            ture_lbl = torch.where(ture>0.5, 1, 0)
            cr = classification_report(ture_lbl.cpu().numpy(), pred_lbl.cpu().numpy(), output_dict=True, zero_division=0)
            clf_report[k] = {
                'acc': cr['accuracy'],
                'precision': cr['macro avg']['precision'],
                'recall': cr['macro avg']['recall'],
                'f1': cr['macro avg']['f1-score']
            }
        bce /= len(outputs.keys())
        mse /= len(outputs.keys())
        rmse = torch.sqrt(mse)
        score = 1/(1+rmse)
        acc, precision, recall, f1 = 0, 0, 0, 0
        for k in clf_report.keys():
            acc += clf_report[k]['acc']
            precision += clf_report[k]['precision']
            recall += clf_report[k]['recall']
            f1 += clf_report[k]['f1']
        acc /= len(clf_report.keys())
        precision /= len(clf_report.keys())
        recall /= len(clf_report.keys())
        f1 /= len(clf_report.keys())

        loss = bce if loss_fn=='BCE' else rmse

        performance = {
            'bce': bce.item(),
            'rmse': rmse.item(),
            'score': score.item(),
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return loss, performance


def get_avg_performance(performance_list):
    avg_performance = {}
    for k in performance_list[0].keys():
        avg_performance[k] = np.mean([performance[k] for performance in performance_list])
    return avg_performance
