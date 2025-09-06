#!/bin/bash

name=$1
ckpt=$2
config=$3

set -a
source .env
set +a

source venv/bin/activate

if [ ! -d "${VoxCeleb1}" ]; then
    echo "Please request access to VoxCeleb1 dataset from https://cn01.mmai.io/keyreq/voxceleb"
    echo "and place it in the path specified by the VoxCeleb1 variable in .env"
    exit 1
fi

cd s3prl
python3 run_downstream.py -m train -n ${name}_asv -u customized_upstream -d sv_voxceleb1 -k ${ckpt} -g ${config} \
    -o "config.downstream_expert.datarc.file_path='${VoxCeleb1}'"
    