#!/bin/bash

echo 'starting up'
env | grep WANDB_API_KEY >> /etc/environment;
env | grep SLACK_HOOK >> /etc/environment;
SLACK_HOOK=$(env | grep SLACK_HOOK | cut -d'=' -f2)


# Setting up workspace
cd /workspace
git clone https://github.com/jackboyla/sentiment-analysis.git
GIT_REPO_PATH=/workspace/sentiment-analysis

# Define the list of config files to train models for
configs=("cloud-canine-backbone.yaml") 

pip install torch kaggle lightning tabulate transformers wandb omegaconf scikit-learn pandas

# Download data
cd $GIT_REPO_PATH
mkdir data
cd data
URL='https://storage.googleapis.com/kaggle-data-sets/2477/4140/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230504%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230504T191741Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=49150b528cec87a1e719d2a8e289fb38bb9883ddf8a088014aef9a6cdc901f4e0ce0cf5ab5b0b6296e02ff7581280095d8da86b5a6be8771dde1cc1a4b3f8b10b7bae7da2bb112a19753c351425e061b6081596e617667237da44ee010f42994a76d8f6bc1a6ac589af8c5a58e3d8091e52ed1014a9564fbe05f4c3cac5ef114111d38b70baed4f494c84e5526e9140a6cb8548f40e3da59f345ec5f04c524f044b361ef780818f3c10032cf2a5a37d120d67a8e676e9f794fbb7adead0e436e3a42263620ffcd121f2aee869642e626da8059579552437d714371ad245cea72d222b5c1059581b8d6c8e14cf6cf4bc1b719e66db63ad46c9adecf6455fb2e52'
wget ${URL} --no-check-certificate -O sentiment140.zip
sudo apt-get install unzip
unzip sentiment140.zip

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

        # Copy the file or directory to the temporary directory
        cp -R "$path" "${TEMP_DIR}/${base_name}"
    done

    # Zip the temporary directory and its contents
    cd "$TEMP_DIR" || return 1
    local DATE_TIME
    DATE_TIME=$(date -u +'%Y-%m-%d__%H-%M-UTC')
    local ZIP_NAME=${DATE_TIME}_${CONFIG}.zip
    zip -r "$ZIP_NAME" .

    # Transfer the zip file using the transfer function
    local TRANSFER_LINK
    TRANSFER_LINK=$(transfer "${TEMP_DIR}/${ZIP_NAME}")

    # Capture the output of ls -1 "$TEMP_DIR" in a variable
    local TEMP_DIR_LS
    TEMP_DIR_LS=$(ls -1 "$TEMP_DIR")

    # Echo a list of all the files and directories that have been zipped
    echo -e "The following files and directories have been zipped: \n${TEMP_DIR_LS}"

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
    BASH_LOG="server_log_${DATE_TIME}_${config%.yaml}.log"
    python src/train.py "configs/${config}" "${BASH_LOG}" &> "${BASH_LOG}"

    # Define the ARTIFACTS you want transferred
    ARTIFACTS=(${GIT_REPO_PATH}/configs/${config})

    # Save CSV logs
    # Get the most recently created directory in logs/lightning_logs/
    latest_dir=$(ls -1t ${GIT_REPO_PATH}/logs/lightning_logs/ | head -n 1)
    ARTIFACTS+=("${GIT_REPO_PATH}/logs/lightning_logs/${latest_dir}")

    # Save model CHECKPOINTS
    latest_dir=$(ls -1t ${GIT_REPO_PATH}/wandb-sentiment-analysis/ | head -n 1)
    ARTIFACTS+=("${GIT_REPO_PATH}/wandb-sentiment-analysis/${latest_dir}")

    # Save wandb log data
    latest_dir=$(ls -1t ${GIT_REPO_PATH}/wandb/ | head -n 1)
    ARTIFACTS+=("${GIT_REPO_PATH}/wandb/${latest_dir}")
    

    zip_and_transfer_artifacts "${config%.yaml}" "${SLACK_HOOK}" "${ARTIFACTS[@]}"
done


# Destroy Instance (from https://vast.ai/faq#Instances)
cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key;

apt-get install -y wget; wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;

./vast destroy instance ${VAST_CONTAINERLABEL:2}

