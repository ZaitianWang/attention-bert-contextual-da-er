# plot train_checkpoint_performance and valid_checkpoint_performance
import matplotlib.pyplot as plt
import time

'''
@Author: Wang
'''

def plot_performance(train_checkpoint_performance, valid_checkpoint_performance):
    # extract ['rmse', 'score', 'acc', 'precision', 'recall', 'f1']
    train_checkpoint_performance = [{k: v for k, v in performance.items() if k in ['rmse', 'score', 'acc', 'precision', 'recall', 'f1']} for performance in train_checkpoint_performance]
    valid_checkpoint_performance = [{k: v for k, v in performance.items() if k in ['rmse', 'score', 'acc', 'precision', 'recall', 'f1']} for performance in valid_checkpoint_performance]
    fig, axs = plt.subplots(2, 3)
    fig.tight_layout(pad=1.0)
    for i, k in enumerate(train_checkpoint_performance[0].keys()):
        train_performance = [performance[k] for performance in train_checkpoint_performance]
        valid_performance = [performance[k] for performance in valid_checkpoint_performance]
        axs[i//3][i%3].plot(train_performance, label='train')
        axs[i//3][i%3].plot(valid_performance, label='valid')
        axs[i//3][i%3].set_title(k)
        axs[i//3][i%3].legend()
    plt.show()
    # name format: vis_log/{time}.png, time format: %d-%H%M
    fig.savefig('vis_log/{}.png'.format(time.strftime("%d-%H%M", time.localtime())))


