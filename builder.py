import torch
import torch.nn as nn
import tqdm
from networks import *

__all__ = ['Model']


class Model(object):
    def __init__(self, num_features, num_labels, network='TransformerMLC', device='cpu', **kwargs):
        if network == "TransformerMLC":
            model = TransformerMLC(num_features=num_features, num_labels=num_labels, **kwargs)
        self.model = model.to(device)
        self.loss_fn = self.set_loss_fn(**kwargs)
        self.optimizer, self.scheduler = self.set_optimizer(**kwargs)

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        scores = self.model(x)
        loss = self.loss_fn(scores, y)
        loss.backward()
        self.optimizer.step()
        pred = torch.sigmoid(scores)
        return loss.item(), pred.data

    def evaluate_step(self, x, y):
        self.model.eval()
        with torch.no_grad():
            scores = self.model(x)
            loss = self.loss_fn(scores, y)
        pred = torch.sigmoid(scores)
        return loss.item(), pred.data

    def train_epoch(self, train_data):
        epoch_loss = 0
        targets = None
        predictions = None
        for i, (x, y) in enumerate(train_data):
            batch_loss, preds = self.train_step(x, y)
            epoch_loss += batch_loss
            if targets is None:
                targets = y
                predictions = preds
            else:
                targets = torch.cat((targets, y), 0)
                predictions = torch.cat((predictions, preds), 0)
        if self.scheduler:
            self.scheduler.step()
        return epoch_loss, targets, predictions

    def evaluate_epoch(self, eval_data):
        epoch_loss = 0
        targets = None
        predictions = None
        for i, (x, y) in enumerate(eval_data):
            batch_loss, preds = self.evaluate_step(x, y)
            epoch_loss += batch_loss
            if targets is None:
                targets = y
                predictions = preds
            else:
                targets = torch.cat((targets, y), 0)
                predictions = torch.cat((predictions, preds), 0)
        return epoch_loss, targets, predictions

    def predict(self, test_data, model_dir):
        self.load_model(model_dir)
        _, predictions = zip(*(self.evaluate_step(x,  y) for x, y in tqdm(test_data, leave=False)))
        return predictions

    def save_model(self, model_dir):
        torch.save(self.model.state_dict(), model_dir)

    def load_model(self, model_dir):
        self.model.load_state_dict(torch.load(model_dir))

    @staticmethod
    def set_loss_fn(loss_fn="bce", **kwargs):
        if loss_fn == "bce":
            fn = nn.BCEWithLogitsLoss()
        return fn

    def set_optimizer(self, betas=(0.9, 0.98), lr=1e-3, step_size=10, gamma=0.1, last_epoch=-1, **kwargs):
        optimizer = torch.optim.Adam(self.model.parameters(), betas=betas, lr=lr)
        scheduler = torch.torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
        return optimizer, scheduler

    @staticmethod
    def seq2bin(seq, num_labels):
        y = torch.zeros((len(seq), num_labels))
        for i in range(len(seq)):
            y[i, seq[i]] = 1
        return y.float()
