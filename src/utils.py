import typing
import importlib
import transformers
import lightning as L
import torch
from typing import List
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_info
import tabulate
import copy
import requests
import datetime
import os

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
        print(f"transformers cache: {os.environ['TRANSFORMERS_CACHE']}")
        model = transformers.AutoModel.from_pretrained(name)
        return model
    else:
        obj_path_list = obj_path.rsplit(".", 1)
        # obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
        obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else obj_path
        obj_name = obj_path_list[0]
        module_obj = importlib.import_module(obj_path)

        if not hasattr(module_obj, obj_name):
            raise AttributeError(
                f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
            )

        return getattr(module_obj, obj_name)
    


class PrintTableMetricsCallback(L.pytorch.callbacks.Callback):
    """
    from (https://stackoverflow.com/questions/40056747/print-a-list-of-dictionaries-in-table-form)
    """

    def __init__(self) -> None:
        self.metrics: List = []
        self.metrics_dict = None

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics_dict = copy.copy(trainer.callback_metrics)
        # rows =  [x.values() for x in metrics_dict]
        self.metrics_dict = metrics_dict

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.metrics_dict:
            val_metrics_dict = copy.copy(trainer.callback_metrics)
            self.metrics_dict.update(val_metrics_dict)
            rows = [self.metrics_dict.values()]
            self.metrics.append(self.metrics_dict)
            rank_zero_info(tabulate.tabulate(rows, self.metrics[0].keys()))

    def on_train_step_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            rank_zero_info("TORCH MEMORY SUMMARY")
            rank_zero_info(torch.cuda.memory_summary())


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
        # Get train loss and metrics from the previous epoch
        epoch_duration = datetime.datetime.now() - self.train_epoch_start_time
        self.train_epoch_duration  = str(epoch_duration - datetime.timedelta(microseconds=epoch_duration.microseconds))
        train_metrics_dict = copy.copy(trainer.callback_metrics)
        self.message_dict['train_loss'] = train_metrics_dict['train_loss']
        self.message_dict['train_F1'] = train_metrics_dict['train_F1']


    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_epoch_start_time = datetime.datetime.now()


    def on_validation_epoch_end(self, trainer, pl_module):
        # Get validation loss and metrics from the current epoch
        if trainer.current_epoch >= 0:
            epoch_duration = datetime.datetime.now() - self.val_epoch_start_time
            self.val_epoch_duration  = str(epoch_duration - datetime.timedelta(microseconds=epoch_duration.microseconds))
            val_metrics_dict = copy.copy(trainer.callback_metrics)
            self.message_dict['val_loss'] = val_metrics_dict['val_loss']
            self.message_dict['val_F1'] = val_metrics_dict['val_F1']

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