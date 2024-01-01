from src.metrics import get_avg_performance


import torch
from tqdm import tqdm


import logging
import time


def do_train(model, train_loader, valid_loader, optimizer, scheduler, criterion, device, train_option):
    EPOCHS = train_option['epochs']
    LOG_STEP = train_option['LOG_STEP']
    LEAST_ACCEPT_SCORE = train_option['LEAST_ACCEPT_SCORE']
    TOY = train_option['TOY']
    '''
    about logs and checkpoints:
    assume log_step = 100
    (1)
    keep a train_batches_performance list every 100 batches
    performance of each batch is appended to the list
    every 100 batches, calculate the average performance of the list and print it, then reset the list
    (2)
    validate every 100 batches
    keep a batch_valid_performance list for all validation batches in the validation set
    after validation, calculate the average performance of the list and print it
    (3)
    avg_performance of each 100-batch-training and 100-batch-validation is appended to train_checkpoint_performance and valid_checkpoint_performance
    train_checkpoint_performance and valid_checkpoint_performance are shared by all epochs, and batching is transparent to them
    (4)
    model checkpoint is saved when the validation performance is better than the previous best
    (5)
    train_checkpoint_performance and valid_checkpoint_performance are used to plot performance curves
    '''
    start_time = time.time()
    logger = logging.getLogger('ABCDE')
    logger.setLevel(logging.INFO)
    # log file name format: day-hour:min.log
    handler = logging.FileHandler(time.strftime('log/%d-%H%M.log'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(handler)
    logger.info('start training')
    train_checkpoint_performance = []
    valid_checkpoint_performance = []
    best_score = LEAST_ACCEPT_SCORE
    global_step = 0
    for epoch in range(EPOCHS):
        train_batches_performance = []
        for train_batch_idx, batch in enumerate(tqdm(train_loader)):
            # train part
            model.train()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = {}
            for k in batch.keys(): # six emotions
                if k not in ['input_ids', 'attention_mask']:
                    labels[k] = batch[k].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss, performance = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_batches_performance.append(performance)
            # validation and log part
            global_step += 1
            if global_step % LOG_STEP == 0:
                avg_performance = get_avg_performance(train_batches_performance)
                train_checkpoint_performance.append(avg_performance)
                # print(f'phase \t bce \t\t rmse \t\t score \t\t acc \t\t precision \t recall \t f1')
                # print(f'train \t {avg_performance["bce"]:0.4f} \t {avg_performance["rmse"]:0.4f} \t {avg_performance["score"]:0.4f} \t {avg_performance["acc"]:0.4f} \t {avg_performance["precision"]:0.4f} \t {avg_performance["recall"]:0.4f} \t {avg_performance["f1"]:0.4f}')
                logger.info(f'phase \t bce \t\t rmse \t\t score \t\t acc \t\t precision \t recall \t f1')
                logger.info(f'train \t {avg_performance["bce"]:0.4f} \t {avg_performance["rmse"]:0.4f} \t {avg_performance["score"]:0.4f} \t {avg_performance["acc"]:0.4f} \t {avg_performance["precision"]:0.4f} \t {avg_performance["recall"]:0.4f} \t {avg_performance["f1"]:0.4f}')
                # contains all performance in the 100 batches, reset it
                train_batches_performance = []
                model.eval()
                batch_valid_performance = []
                with torch.no_grad():
                    for val_batch_idx, batch in enumerate(valid_loader):
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = {}
                        for k in batch.keys():
                            if k not in ['input_ids', 'attention_mask']:
                                labels[k] = batch[k].to(device)
                        outputs = model(input_ids, attention_mask)
                        loss, performance = criterion(outputs, labels)
                        batch_valid_performance.append(performance)
                avg_performance = get_avg_performance(batch_valid_performance)
                valid_checkpoint_performance.append(avg_performance)
                # print(f'valid \t {avg_performance["bce"]:0.4f} \t {avg_performance["rmse"]:0.4f} \t {avg_performance["score"]:0.4f} \t {avg_performance["acc"]:0.4f} \t {avg_performance["precision"]:0.4f} \t {avg_performance["recall"]:0.4f} \t {avg_performance["f1"]:0.4f}')
                logger.info(f'valid \t {avg_performance["bce"]:0.4f} \t {avg_performance["rmse"]:0.4f} \t {avg_performance["score"]:0.4f} \t {avg_performance["acc"]:0.4f} \t {avg_performance["precision"]:0.4f} \t {avg_performance["recall"]:0.4f} \t {avg_performance["f1"]:0.4f}')
                # save best model
                if avg_performance['score'] > best_score:
                    best_score = avg_performance['score']
                    # name format: weights/{score}-{day}-{hour:min}-{epoch}-{batch_idx}.pt, score format: 0.0000
                    torch.save(model, 'weights/{:.4f}-{}-{}-{}-{}.pt'.format(best_score, time.strftime("%d", time.localtime()), time.strftime("%H%M", time.localtime()), epoch, train_batch_idx))
                    if TOY and len(valid_checkpoint_performance) > 1: # to see if code works; ignore/delete in final release
                        break # toy train, break after first best model is derived and at least two checkpoints for ploting
    logger.info('end training')
    logger.info('best score: {} at epoch {} batch {}'.format(best_score, epoch, train_batch_idx))
    logger.info('time elapsed: {}s'.format(time.time() - start_time))
    return train_checkpoint_performance, valid_checkpoint_performance
