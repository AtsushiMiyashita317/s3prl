#!/bin/bash

name=$1
ckpt=$2
config=$3

set -a
source .env
set +a

source venv/bin/activate

if [ ! -d "${SpeechCommands}/train" ]; then
    echo "Downloading and extracting Speech Commands dataset..."
    wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
    mkdir -p ${SpeechCommands}/train
    tar -xvzf speech_commands_v0.01.tar.gz -C ${SpeechCommands}/train
    rm speech_commands_v0.01.tar.gz
fi

if [ ! -d "${SpeechCommands}/test" ]; then
    echo "Downloading and extracting Speech Commands test dataset..."
    wget -c http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz
    mkdir -p ${SpeechCommands}/test
    tar -xvzf speech_commands_test_set_v0.01.tar.gz -C ${SpeechCommands}/test
    rm speech_commands_test_set_v0.01.tar.gz
fi

cd s3prl
python3 run_downstream.py -m train -n ${name}_ks -u customized_upstream -d speech_commands -k ${ckpt} -g ${config} \
    -o "config.downstream_expert.datarc.speech_commands_root='${SpeechCommands}/train',,config.downstream_expert.datarc.speech_commands_test_root='${SpeechCommands}/test'"
