#!/bin/bash

# Record the start time
start_time=$(date +%s)

echo 'starting up'
env | grep SLACK_HOOK >> /etc/environment;
SLACK_HOOK=$(env | grep SLACK_HOOK | cut -d'=' -f2)

# Setting up workspace
cd /workspace
git clone https://github.com/jackboyla/sentiment-analysis.git
GIT_REPO_PATH=/workspace/sentiment-analysis

# Define the list of config files to train models for
configs=("canine-backbone.yaml") 

pip install torch kaggle lightning tabulate transformers wandb omegaconf scikit-learn pandas gdown


# Download data
cd $GIT_REPO_PATH
gdown --fuzzy 'https://drive.google.com/drive/folders/1qZ1yguFcr6VGA-rrgrxtdjkb0WLXBcQO?usp=sharing' -O ./data --folder
sudo apt-get install -y zip
sudo apt-get install -y unzip


# https://github.com/dutchcoders/transfer.sh/blob/main/examples.md#using-curl
transfer() {
    curl --progress-bar --upload-file "$1" https://transfer.sh/$(basename "$1") | tee /dev/null;
    echo
}
alias transfer=transfer


# http://blog.pragbits.com/it/2015/02/09/slack-notifications-via-curl/
function post_to_slack () {
  # format message as a code block ```${msg}```
  SLACK_MESSAGE="\`\`\`$1\`\`\`"
  SLACK_URL="$3"
 
  case "$2" in
    INFO)
      SLACK_ICON=':slack:'
      ;;
    WARNING)
      SLACK_ICON=':warning:'
      ;;
    ERROR)
      SLACK_ICON=':bangbang:'
      ;;
    SAVE)
      SLACK_ICON=':floppy_disk:'
      ;;
  esac
 
  curl -X POST --data "payload={\"text\": \"${SLACK_ICON} ${SLACK_MESSAGE}\"}" ${SLACK_URL}
}


# Move all files and dirs in ARTIFACTS into a temp dir
# Zip the temp dir, transfer.sh the files and post a message to Slack
function zip_and_transfer_artifacts() {
    local CONFIG="$1"
    local SLACK_HOOK="$2"
    shift 2
    local ARTIFACTS=("$@")
    local TEMP_DIR

    TEMP_DIR=$(mktemp -d)

    # Move all files and directories to the temporary directory
    for path in "${ARTIFACTS[@]}"; do
        # Get the base name of the path
        base_name=$(basename "$path")

        # Move the file or directory to the temporary directory
        mv -R "$path" "${TEMP_DIR}/${base_name}"
    done

    # Zip the temporary directory and its contents
    cd "$TEMP_DIR" || return 1
    local DATE_TIME
    DATE_TIME=$(date -u +'%Y-%m-%d__%H-%M-UTC')
    local ZIP_NAME=${DATE_TIME}_${CONFIG}.zip
    zip -r "$ZIP_NAME" .

    # Capture the output of ls -1 "$TEMP_DIR" in a variable
    local TEMP_DIR_LS
    TEMP_DIR_LS=$(ls -1 "$TEMP_DIR")

    # Echo a list of all the files and directories that have been zipped
    echo -e "The following files and directories have been zipped: \n${TEMP_DIR_LS}"

    # Record the start time
    transfer_start_time=$(date +%s)

    # Transfer the zip file using the transfer function
    local TRANSFER_LINK
    TRANSFER_LINK=$(transfer "${TEMP_DIR}/${ZIP_NAME}")

    # Record the end time
    transfer_end_time=$(date +%s)

    # Calculate the elapsed time
    transfer_elapsed_time=$(($transfer_end_time - $transfer_start_time))

    # Echo the elapsed time
    echo "Transfer Elapsed time: $transfer_elapsed_time seconds"

    # Remove the temporary directory and zip file
    rm -rf "$TEMP_DIR"

    # Display the transfer link
    echo "TRANSFER_LINK: ${TRANSFER_LINK}"

    post_to_slack "Files saved: \n${TEMP_DIR_LS} \nTRANSFER_LINK: ${TRANSFER_LINK}" "SAVE" "${SLACK_HOOK}"
}


cd $GIT_REPO_PATH

# Loop through the config files
for config in "${configs[@]}"
do
    # Get the current date and time
    DATE_TIME=$(date -u +'%Y-%m-%d__%H-%M-UTC')
    
    # Run the train.py script with the current config file and send output to logs
    echo "Training Starting..."
    BASH_LOG="server_log_${DATE_TIME}_${config%.yaml}.log"
    python src/train.py --cfg_path "configs/${config}" --server_log_file "${BASH_LOG}" &> "${BASH_LOG}"

    # Define the ARTIFACTS you want transferred

    # Save the training logs
    ARTIFACTS=(${GIT_REPO_PATH}/${BASH_LOG})

    # Save model CHECKPOINTS
    latest_dir=$(ls -1t ${GIT_REPO_PATH}/wandb-sentiment-analysis/ | head -n 1)
    ARTIFACTS+=("${GIT_REPO_PATH}/wandb-sentiment-analysis/${latest_dir}")

    # Save wandb log data
    latest_dir=$(ls -1t ${GIT_REPO_PATH}/wandb/ | head -n 1)
    ARTIFACTS+=("${GIT_REPO_PATH}/wandb/${latest_dir}")
    
    # Zip, Transfer and Remove Relevant files/dirs
    zip_and_transfer_artifacts "${config%.yaml}" "${SLACK_HOOK}" "${ARTIFACTS[@]}"
done


# Wait for all background processes to finish before destroying the instance
wait


# Record the end time
end_time=$(date +%s)

# Calculate the elapsed time
elapsed_time=$(($end_time - $start_time))

# Echo the elapsed time
echo "Elapsed time: $elapsed_time seconds"

post_to_slack "Script finishing! Elapsed Time: ${elapsed_time}" "INFO" "${SLACK_HOOK}"


# Destroy Instance (from https://vast.ai/faq#Instances)
cd /

cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > /ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > /instance_id_hv; head -c -1 -q /ssh_key_hv /instance_id_hv > ~/.vast_api_key;

sudo apt-get install -y wget; wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;

./vast destroy instance ${VAST_CONTAINERLABEL:2}

