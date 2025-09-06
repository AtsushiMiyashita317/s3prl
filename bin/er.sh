#!/bin/bash

name=$1
ckpt=$2
config=$3

set -a
source .env
set +a

if [ ! -d "${IEMOCAP}" ]; then
    echo "Please request access to IEMOCAP dataset from https://sail.usc.edu/iemocap/"
    echo "and place it in the path specified by the IEMOCAP variable in .env"
    exit 1
fi

cd s3prl
python3 run_downstream.py -m train -n ${name}_er -u customized_upstream -d emotion -k ${ckpt} -g ${config} -f \
    -o "config.downstream_expert.datarc.root='${IEMOCAP}'"
