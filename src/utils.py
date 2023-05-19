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

import sys
import logging


def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s: %(message)s'
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

# https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
def load_obj(obj_path: str, default_obj_path: str = "", name: str = None) -> typing.Any:
    
    """
    Used to Load Objects from config files.
    Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """

    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    # obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)

    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
        )

    return getattr(module_obj, obj_name)
    


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