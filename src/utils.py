import typing
import importlib
from transformers import AutoModel

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
    
    # TO-DO: FIND MORE ELEGANT SOLUTION
    if 'transformers' in obj_path:
        model = AutoModel.from_pretrained(name)
        return model
    else:
        obj_path_list = obj_path.rsplit(".", 1)
        obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
        obj_name = obj_path_list[0]
        module_obj = importlib.import_module(obj_path)

        if not hasattr(module_obj, obj_name):
            raise AttributeError(
                f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
            )

        return getattr(module_obj, obj_name)
    

import lightning as L
from typing import List
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_info
import tabulate
import copy

class PrintTableMetricsCallback(L.pytorch.callbacks.Callback):
    """
    from (https://stackoverflow.com/questions/40056747/print-a-list-
    of-dictionaries-in-table-form)
    """

    def __init__(self) -> None:
        self.metrics: List = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics_dict = copy.copy(trainer.callback_metrics)
        rows =  [x.values() for x in metrics_dict]
        self.metrics.append(metrics_dict)
        rank_zero_info(tabulate.tabulate(rows, self.metrics[0].keys()))





import requests
import datetime
import lightning as L

class SlackCallback(L.pytorch.callbacks.Callback):
    def __init__(self, webhook_url, cfg, server_log_file):
        super().__init__()
        self.webhook_url = webhook_url
        self.message_dict = {}
        self.train_epoch_duration = None
        self.val_epoch_duration = None
        self.cfg = cfg
        self.server_log_file = server_log_file
    
    def on_train_start(self, trainer, pl_module):
        attachment = [
            {
                "color": "#36a64f",
                "text": f"Config:\n{self.cfg}"
            }
        ]
        payload = {
            "attachments": attachment,
            "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f":zap:Training Started!:hugging_face: `{self.server_log_file}`",
                            "emoji": True
                        }
                    },
                ]

        }
        self.post_to_slack(payload)

    
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_start_time = datetime.datetime.now()
            

    def on_train_epoch_end(self, trainer, pl_module):
        # Get train loss and metrics from the previous epoch
        epoch_duration = datetime.datetime.now() - self.train_epoch_start_time
        self.train_epoch_duration  = str(epoch_duration - datetime.timedelta(microseconds=epoch_duration.microseconds))
        self.message_dict['train_loss'] = trainer.callback_metrics['train_loss']
        self.message_dict['train_F1'] = trainer.callback_metrics['train_F1']


    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_epoch_start_time = datetime.datetime.now()


    def on_validation_epoch_end(self, trainer, pl_module):
        # Get validation loss and metrics from the current epoch
        if trainer.current_epoch > 0:
            epoch_duration = datetime.datetime.now() - self.val_epoch_start_time
            self.val_epoch_duration  = str(epoch_duration - datetime.timedelta(microseconds=epoch_duration.microseconds))
            self.message_dict['val_loss'] = trainer.callback_metrics['val_loss']
            self.message_dict['val_F1'] = trainer.callback_metrics['val_F1']

            payload = self.format_message_dict(trainer.current_epoch)

            # Send the message to Slack using the webhook URL
            self.post_to_slack(payload)


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