import torch
import logging
import logging.config
import sys
import time
import os
from sklearn.metrics import accuracy_score, hamming_loss, f1_score

__all__ = ['run']


def run(train_data, test_data, model, opt):
    if not os.path.exists(opt['results_dir']):
        os.makedirs(opt['results_dir'])

    logger = get_logger(opt['name'], opt['results_dir'])

    logger.info("Here is your model for MLC:")
    logger.info(model.model)

    logger.info("{t} - Training starts".format(t=time.strftime('%H:%M:%S')))

    logger.info("{a}".format(a=model.device))
    i, x, y = next(iter(train_data))
    logger.info("{a}".format(a=x.device))
    logger.info("{a}".format(a=y.device))

    train_loss = []
    test_loss = []
    best_epoch_score = 0
    for epoch in range(opt['num_epochs']):
        epoch_loss, targets, predictions = model.train_epoch(train_data)
        train_loss.append(epoch_loss)
        _, epoch_score, threshold = compute_metrics(predictions, targets)
        logger.info("Epoch {:} - average training loss = {:.6}".format(epoch, epoch_loss/len(predictions)))

        if epoch % opt['test_step'] == 0:
            epoch_loss, targets, predictions = model.evaluate_epoch(test_data)
            test_loss.append(epoch_loss)
            performance, _, _ = compute_metrics(predictions, targets, threshold)
            logger.info("Epoch {:} - average test loss = {:.6}".format(epoch, epoch_loss / len(predictions)))
            logger.info("Calculated performance metrics on test data")
            log_performance(logger, performance)

        if epoch_score > best_epoch_score:
            model.save_model(opt['results_dir']+opt['name']+'.model')
            best_epoch_score = epoch_score

    logger.info("Final test results with best model")
    epoch_loss, targets, predictions = model.evaluate_epoch(test_data)
    performance, _, _ = compute_metrics(predictions, targets)
    log_performance(logger, performance)

    torch.save(torch.as_tensor(train_loss, dtype=float), opt['results_dir'] + opt['name']+'train_losses.pt')
    torch.save(torch.as_tensor(test_loss, dtype=float), opt['results_dir'] + opt['name']+'test_losses.pt')


def compute_metrics(predictions, targets, br_threshold=None):
    targets = targets.detach().numpy()
    predictions = predictions.detach().numpy()

    def _calculate(level, p=predictions.copy(), t=targets):
        p[p < level] = 0
        p[p >= level] = 1
        acc = accuracy_score(t, p)
        ha = 1 - hamming_loss(t, p)
        ebf1 = f1_score(t, p, average='samples')
        mif1 = f1_score(t, p, average='micro')
        maf1 = f1_score(t, p, average='macro')
        return acc, ha, ebf1, mif1, maf1

    best_score = 0
    if br_threshold is None:
        for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            ACC, HA, ebF1, miF1, maF1 = _calculate(tau)
            score = ACC + HA + ebF1 + miF1 + maF1
            if score >= best_score:
                br_threshold = tau
                best_score = score

    ACC, HA, ebF1, miF1, maF1 = _calculate(br_threshold)
    metrics = {'ACC': ACC, 'HA': HA, 'ebF1': ebF1, 'miF1': miF1, 'maF1': maF1}

    return metrics, best_score,  br_threshold


def log_performance(logger, metrics):
    logger.info('ACC:    \t {:.6} '.format(metrics['ACC']))
    logger.info('HA:     \t {:.6} '.format(metrics['HA']))
    logger.info('ebF1:   \t {:.6} '.format(metrics['ebF1']))
    logger.info('miF1:   \t {:.6} '.format(metrics['miF1']))
    logger.info('maF1:   \t {:.6} '.format(metrics['maF1']))


def get_logger(name, log_dir='./results/'):
    name = name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')

    config_dict = {"version": 1,
                   "formatters": {"base": {"format": "%(message)s"}},
                   "handlers": {"base": {"class": "logging.FileHandler",
                                         "level": "DEBUG",
                                         "formatter": "base",
                                         "filename": log_dir + name,
                                         "encoding": "utf8"}},
                   "root": {"level": "DEBUG", "handlers": ["base"]},
                   "disable_existing_loggers": False}
    logging.config.dictConfig(config_dict)

    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger

