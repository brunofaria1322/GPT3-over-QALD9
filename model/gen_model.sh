#!/bin/sh

DATA_PATH="./"
DATA_FILE_NAME="ft-qald9_train"
EXTENSION="jsonl"
SUFFIX="_prepared"

# models available: ada, babbage, curie, davinci
MODEL="davinci"

# echo text from file
echo "Generating model from $DATA_PATH$DATA_FILE_NAME.$EXTENSION"
openai tools fine_tunes.prepare_data -f "$DATA_PATH$DATA_FILE_NAME.$EXTENSION" -q

# set openai api key from text file
export OPENAI_API_KEY=$(cat ../data/key.txt)
echo "Using API key: $OPENAI_API_KEY"

# fine tune model
echo "Fine tuning model"
openai api fine_tunes.create -t "$DATA_PATH$DATA_FILE_NAME$SUFFIX.$EXTENSION" -m "$MODEL"
#openai api fine_tunes.create -t "./ft-qald9_train_prepared.jsonl" -m "davinci"

# RESUME IF INTERRUPTED
#openai api fine_tunes.follow -i ft-n0rQ075rM5Psqt9EONbaSe4o

# GENERATED MODEL
# openai api completions.create -m davinci:ft-personal-2022-12-02-20-52-35 -p <YOUR_PROMPT>
