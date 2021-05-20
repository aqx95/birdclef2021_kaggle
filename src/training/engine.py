import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import datetime
import time
import pytz
import torch
import logging

from meter import MetricMeter
from utils.logger import log
from sklearn.metrics import label_ranking_average_precision_score


def mixup_data(x, y, alpha=0.4):
    """
    Applies mixup to a sample
    Arguments:
        x {torch tensor} -- Input batch
        y {torch tensor} -- Labels
    Keyword Arguments:
        alpha {float} -- Parameter of the beta distribution (default: {0.4})
    Returns:
        torch tensor  -- Mixed input
        torch tensor  -- Labels of the original batch
        torch tensor  -- Labels of the shuffle batch
        float  -- Probability samples by the beta distribution
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


class Fitter():
    def __init__(self, model, device, config, class_weight):
        self.model = model
        self.device = device
        self.config = config
        self.logger = logging.getLogger('training')

        self.epoch = 0
        self.best_f1 = 0
        self.track_train = {"loss": [], "lrap":[], "precision":[], "recall":[], "f1":[]}
        self.track_valid = {"loss": [], "lrap":[], "precision":[], "recall":[], "f1":[]}

        if not os.path.exists(self.config.SAVE_PATH):
            os.makedirs(self.config.SAVE_PATH)
        if not os.path.exists(self.config.LOG_PATH):
            os.makedirs(self.config.LOG_PATH)

        if config.USE_WEIGHT:
            self.loss = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        else:
            self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=8e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=1e-5, T_max=self.config.NUM_EPOCHS)


    def fit(self, train_loader, valid_loader, fold):
        self.logger.info("Training on Fold {} with {}".format(fold, self.config.MODEL_NAME))

        for epoch in range(self.config.NUM_EPOCHS):
            #Training
            start_time = time.time()
            train_metrics = self.train_one_epoch(train_loader)
            end_time = time.time()

            for key, value in train_metrics.items():
                self.track_train[key].append(value)

            train_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
            self.logger.info("[RESULT]: Train. Epoch {} | Loss: {:.3f} | LRAP: {:.3f} | " \
                    "F1: {:.3f} | Precision: {:.3f} | Recall: {:.3f} | " \
                    "Time Elapsed: {}".format(self.epoch, train_metrics['loss'], train_metrics['lrap'],
                    train_metrics['f1'], train_metrics['precision'], train_metrics['recall'], train_elapsed_time))

            #Validation
            start_time = time.time()
            valid_metrics = self.validate_one_epoch(valid_loader)
            end_time = time.time()

            for key, value in valid_metrics.items():
                self.track_valid[key].append(value)

            valid_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
            self.logger.info("[RESULT]: Validate. Epoch {} | Loss: {:.3f} | LRAP: {:.3f} | "\
                    "F1: {:.3f} | Precision: {:.3f} | Recall: {:.3f} | "\
                    "Time Elapsed: {}".format(self.epoch, valid_metrics['loss'], valid_metrics['lrap'],
                    valid_metrics['f1'], valid_metrics['precision'], valid_metrics['recall'], valid_elapsed_time))

            self.monitored_metrics = valid_metrics['f1']
            #Save
            if self.best_f1 < valid_metrics['f1']:
              self.logger.info(f"F1 score improved {self.best_f1} -> {valid_metrics['f1']}. Saving Model...")
              self.save(os.path.join(self.config.SAVE_PATH, '{}_fold{}.pt'.format(self.config.MODEL_NAME, fold)))
              self.best_f1 = valid_metrics['f1']

            #Update scheduler
            if self.config.VAL_STEP_SCHEDULER:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.monitored_metrics)
                else:
                    self.scheduler.step()

            self.epoch += 1
            self.logger.info("----------------------------------------------------------------")

        return self.track_train, self.track_valid


    def train_one_epoch(self, train_loader):
        self.model.train()
        meter = MetricMeter()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (imgs, labels) in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            batch_size = labels.shape[0]
            metric = {}

            if np.random.rand() < 0.5 and self.config.MIXUP:
                imgs, y_a, y_b, _ = mixup_data(imgs, labels, alpha=5)
                labels = torch.clamp(y_a + y_b, 0, 1)

            self.optimizer.zero_grad()
            output = self.model(imgs)
            loss = self.loss(output, labels)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                metric['loss'] = loss.item()
                output = output.sigmoid()
                labels = (labels > 0.5) * 1.0
                metric['lrap'] = label_ranking_average_precision_score(labels.cpu().numpy(), output.cpu().numpy())

                output = (output > 0.5) * 1.0
                precision = (output * labels).sum() / (1e-6 + output.sum())
                metric['precision'] = precision.item()
                recall = (output * labels).sum() / (1e-6 + labels.sum())
                metric['recall'] = recall.item()
                f1 = (2 * precision * recall) / (1e-6 + precision + recall)
                metric['f1'] = f1.item()

            meter.update(metric)

            if self.config.TRAIN_STEP_SCHEDULER:
                self.scheduler.step(self.epoch + step/len(train_loader))

            metrics = meter.avg
            if self.config.VERBOSE:
                pbar.set_postfix(loss = f"{metrics['loss']:.3f}",
                                 lrap = f"{metrics['lrap']:.3f}",
                                 precision = f"{metrics['precision']:.3f}",
                                 recall = f"{metrics['recall']:.3f}",
                                 f1 = f"{metrics['f1']:.3f}")

        return meter.avg


    def validate_one_epoch(self, valid_loader):
        self.model.eval()
        pred_list, label_list = [],[]
        start_time = time.time()
        metric = {}

        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        with torch.no_grad():
            for step, (imgs, labels) in pbar:
                imgs, labels= imgs.to(self.device), labels.to(self.device)

                prediction = self.model(imgs)
                label_list.append(labels)
                pred_list.append(prediction)

            pred_list, label_list = torch.cat(pred_list), torch.cat(label_list)
            loss = self.loss(pred_list, label_list).item()
            metric['loss'] = loss

            pred = pred_list.sigmoid()
            labels = (label_list > 0.5) * 1.0
            lrap = label_ranking_average_precision_score(labels.cpu().numpy(), pred.cpu().numpy())
            metric['lrap'] = lrap

            pred = (pred > 0.5) * 1.0
            precision = ((pred * labels).sum() / (1e-6 + pred.sum())).item()
            metric['precision'] = precision
            recall = ((pred * labels).sum() / (1e-6 + labels.sum())).item()
            metric['recall'] = recall
            f1 = (2 * precision * recall) / (1e-6 + precision + recall)
            metric['f1'] = f1

        return metric


    def save(self, path):
      self.model.eval()
      torch.save(
          {
              "model_state_dict": self.model.state_dict(),
              "optimizer_state_dict": self.optimizer.state_dict(),
              "scheduler_state_dict": self.scheduler.state_dict(),
              "best_f1": self.best_f1,
              "epoch": self.epoch,
          },
          path
      )
