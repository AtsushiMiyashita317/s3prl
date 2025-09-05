#!/bin/bash

name=$1
ckpt=$2
config=$3

set -a
source .env
set +a

if [ ! -d "${FluentSpeechCommands}" ]; then
    echo "Please download FluentSpeechCommands dataset from https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus and extract it to ${FluentSpeechCommands} directory."
    exit 1
fi

cd s3prl
python3 run_downstream.py -m train -n ${name}_ic -u customized_upstream -d fluent_commands -k ${ckpt} -g ${config} \
    -o "config.downstream_expert.datarc.file_path='${FluentSpeechCommands}'"
