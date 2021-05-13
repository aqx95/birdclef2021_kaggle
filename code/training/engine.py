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

class Fitter():
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.logger = logging.getLogger('training')

        self.epoch = 0

        if not os.path.exists(self.config.SAVE_PATH):
            os.makedirs(self.config.SAVE_PATH)
        if not os.path.exists(self.config.LOG_PATH):
            os.makedirs(self.config.LOG_PATH)

        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=8e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=self.config.NUM_EPOCHS)


    def fit(self, train_loader, valid_loader, fold):
        self.logger.info("Training on Fold {} with {}".format(fold, self.config.MODEL_NAME))

        for epoch in range(self.config.NUM_EPOCHS):
            #Training
            start_time = time.time()
            train_metrics = train_one_epoch(train_loader)
            end_time = time.time()

            train_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
            self.logger.info("[RESULT]: Train. Epoch {} | Loss: {:.3f} | LRAP: {:.3f} | \
                    F1: {:.3f} | Precision: {:.3f} | Recall: {:.3f} \
                    Time Elapsed: {}".format(self.epoch, train_metrics['loss'], train_metrics['lrap'],
                    train_metrics['f1'], train_metrics['precision'], train_metrics['recall'], train_elapsed_time))

            #Validation
            start_time = time.time()
            valid_metrics = validate_one_epoch(train_loader)
            end_time = time.time()

            valid_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
            self.logger.info("[RESULT]: Validate. Epoch {} | Loss: {:.3f} | LRAP: {:.3f} | \
                    F1: {:.3f} | Precision: {:.3f} | Recall: {:.3f} \
                    Time Elapsed: {}".format(self.epoch, valid_metrics['loss'], valid_metrics['lrap'],
                    valid_metrics['f1'], valid_metrics['precision'], valid_metrics['recall'], valid_elapsed_time))

            #update scheduler
            if self.config.val_step_scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.monitored_metrics)
                else:
                    self.scheduler.step()

            self.epoch += 1


    def train_one_epoch(self, train_loader):
        self.model.train
        meter = MetricMeter()
        start_time = time.time()

        pbar = tqdm(enumerate(train_loader), len(train_loader))
        for step, (imgs, labels) in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            batch_size = labels.shape[0]
            metric = {}

            self.optimizer.zero_grad()
            output = self.model(imgs)
            loss = self.loss(output, labels)
            loss.backward()
            optimizer.step()
            metric['loss'] = loss.item()

            with torch.no_grad():
                output = output.sigmoid()
                labels = (labels > 0.5) * 1.0
                metric['lrap'] = label_ranking_average_precision_score(labels.cpu().numpy(), output.cpu().numpy())

                output = (output > 0.5) * 1.0
                precision = (output * labels).sum() / (1e-6 + output.sum())
                metric['precision'] = precision
                recall = (output * labels).sum() / (1e-6 + labels.sum())
                metric['recall'] = recall
                f1 = (2 * precision * recall) / (1e-6 + precision + recall)
                metric['f1'] = f1

            meter.update(metric, batch_size)

            if self.config.TRAIN_STEP_SCHEDULER:
                self.scheduler.step(self.epoch + step/len(train_loader))

            end_time = time.time()
            metrics = meter.avg
            if self.config.VERBOSE:
                description = f"Train Steps {step}/{len(train_loader)} loss: {metrics['loss']:.3f}, \
                                lrap: {metrics['lrap']:.3f}, precision: {metrics['precision']:.3f}, \
                                recall: {metrics['recall']:.3f}, f1: {metrics['f1']:.3f} \
                                time: {(end_time - start_time):.3f}"
                pbar.set_description(description)

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
            precision = ((pred * labels).sum() / (1e-6 + output.sum())).item()
            metric['precision'] = precision
            recall = ((pred * labels).sum() / (1e-6 + labels.sum())).item()
            metric['recall'] = recall
            f1 = 2 * precision * recall
            metric['f1'] = f1

        return metric
