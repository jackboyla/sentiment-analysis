
import typing
import importlib
import lightning as L
from typing import List
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_info
import tabulate
import copy
import requests
import datetime
import os
import torch

import sys
import logging


class PrintTableMetricsCallback(L.pytorch.callbacks.Callback):
    """
    Print a table of *epoch-level* metrics
    Refer to here: https://github.com/Lightning-AI/lightning/discussions/7722#discussioncomment-787435
    for info on the difference between values sent to trainer.callback_metrics and trainer.logger_metrics
    
    (Tabulate: https://stackoverflow.com/questions/40056747/print-a-list-of-dictionaries-in-table-form)
    """

    def __init__(self) -> None:
        self.metrics: List = []
        self.metrics_dict = None

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics_dict = copy.copy(trainer.callback_metrics)
        # rows =  [x.values() for x in metrics_dict]
        self.metrics_dict = metrics_dict
        self.metrics_dict['epoch'] = trainer.current_epoch
        rows = [self.metrics_dict.values()]
        self.metrics.append(self.metrics_dict)
        rank_zero_info(tabulate.tabulate(rows, self.metrics[0].keys()))



class InputMonitor(L.pytorch.callbacks.Callback):

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            logger = trainer.logger
            logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            logger.experiment.add_histogram("target", y, global_step=trainer.global_step)


class CheckBatchGradient(L.pytorch.callbacks.Callback):
    
    def on_train_start(self, trainer, model):
        n = 0

        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()
        
        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)
        
        if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")
        

class VanishingGradientCallback(L.pytorch.callbacks.Callback):
    def __init__(self, threshold=1e-5):
        super().__init__()
        self.threshold = threshold

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Iterate over the parameters and check their gradient norms
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm < self.threshold:
                    rank_zero_info(f"Vanishing gradient detected in parameter '{name}': {grad_norm}")
        
        if torch.isnan(outputs['loss']):
            trainer.should_stop = True
            rank_zero_info("Training stopped due to NaN loss.")


class SlackCallback(L.pytorch.callbacks.Callback):
    def __init__(self, cfg, server_log_file):
        super().__init__()
        self.webhook_url = os.environ['SLACK_HOOK']
        self.message_dict = {}
        self.train_epoch_duration = None
        self.val_epoch_duration = None
        self.cfg = cfg
        self.server_log_file = server_log_file
    
    def on_train_start(self, trainer, pl_module):
        self.train_epoch_start_time = datetime.datetime.now()
        attachment = [
            {
                "color": "#36a64f",
                "text": f"Run at : `{self.server_log_file}`\nConfig:\n{self.cfg}"
            }
        ]
        payload = {
            "attachments": attachment,
            "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f":zap:Training Started!:hugging_face:",
                            "emoji": True
                        }
                    },
                ]

        }
        self.post_to_slack(payload)


    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_start_time = datetime.datetime.now()
            

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.state.stage != 'sanity_check':
            self.train_epoch_duration = datetime.datetime.now() - self.train_epoch_start_time
            self.train_epoch_duration = str(self.train_epoch_duration - datetime.timedelta(microseconds=self.train_epoch_duration.microseconds))
            
            metrics_dict = copy.copy(trainer.callback_metrics)
            self.message_dict['train_loss'] = metrics_dict['train_loss']
            self.message_dict['train_F1'] = metrics_dict['train_F1']
            # self.message_dict['train_acc'] = metrics_dict['train_acc']
            self.message_dict['val_loss'] = metrics_dict['val_loss']
            self.message_dict['val_F1'] = metrics_dict['val_F1']
            # self.message_dict['val_acc'] = metrics_dict['val_acc']

            payload = self.format_message_dict(trainer.current_epoch)

            # Send the message to Slack using the webhook URL
            self.post_to_slack(payload)


    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_epoch_start_time = datetime.datetime.now()


    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_epoch_duration = datetime.datetime.now() - self.val_epoch_start_time
        self.val_epoch_duration  = str(self.val_epoch_duration  - datetime.timedelta(microseconds=self.val_epoch_duration.microseconds))


    def on_train_end(self, trainer, pl_module):
        # Notify on train end
        payload = {
            "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": ":white_check_mark: Training Finished! :alien:",
                            "emoji": True
                        }
                    },
                ]

        }
        self.post_to_slack(payload)


    def post_to_slack(self, payload):
        requests.post(self.webhook_url, json=payload)


    def format_message_dict(self, current_epoch):
        message = f""
        if self.train_epoch_duration:
            message += f">:robot_face: Train epoch completed in `{self.train_epoch_duration}`\n>"
        if self.val_epoch_duration:
            message += f"\n>:innocent: Validation epoch completed in `{self.val_epoch_duration}`\n"

        message += ":dart: Metrics"
        for tracked_item, tracked_value in self.message_dict.items():
            message += "\n"
            message += f"{tracked_item}: ```{tracked_value:.3f}```"

        # Define Slack message formatting options
        payload = {
            "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"Epoch {current_epoch}",
                            "emoji": True
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": message
                        }
                    },
                    {
                        "type": "divider"
                    }
                ]

        }

        return payload