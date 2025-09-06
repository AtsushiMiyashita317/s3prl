#!/bin/bash

name=$1
ckpt=$2
config=$3

set -a
source .env
set +a

source venv/bin/activate

if [ ! -d "${Quesst14}" ]; then
    echo "Downloading and extracting Quesst14 dataset..."
    wget https://speech.fit.vutbr.cz/files/quesst14Database.tgz
    mkdir -p ${Quesst14}
    tar -xvzf quesst14Database.tgz -C ${Quesst14}/../
    rm quesst14Database.tgz
fi

cd s3prl
python3 run_downstream.py -m train -n ${name}_qbe -u customized_upstream -d quesst14_embedding -k ${ckpt} -g ${config} \
    -o "config.downstream_expert.datarc.quesst2014_root='${Quesst14}'"
    