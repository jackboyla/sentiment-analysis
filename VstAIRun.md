
# Run Training Using Vast AI
Assuming you already have a Vast.ai instance set up, follow these steps to train a model:

1. Inside your Vast.ai console, choose an image  
2. Add environment variables to the image like so:
```-e SLACK_HOOK=... -e WANDB_API_KEY=...```

2. Copy the contents of `vast-ai-run.sh` into the startup script. Feel free to modify the script to suit your needs.
3. Rent an instance and training will commence automatically. 

Slack will provide updates if the callback is enabled in the config YAML. WandB will run information provided that the logger is enabled in the config YAML. To view the logs on the WandB dashboard, a WANDB_API_KEY must be provided as shown above.

The instance will run src/train.py with all specified config files. After each training run finishes, the files are uploaded via transfer.sh, where they will be available for download. The instance will then destroy itself to avoid wasting resources.

The transfer.sh link will be provided via Slack.
