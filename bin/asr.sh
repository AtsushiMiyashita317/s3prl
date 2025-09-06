#!/bin/bash

name=$1
ckpt=$2
config=$3

set -a
source .env
set +a

subsets=(
    dev-clean
    test-clean
    train-clean-100
)

for subset in "${subsets[@]}"; do
    if [ ! -d "${LibriSpeech}/${subset}" ]; then
        echo "Downloading and extracting ${subset}..."
        wget http://www.openslr.org/resources/12/${subset}.tar.gz
        tar -xvzf ${subset}.tar.gz -C ${LibriSpeech}/..
        rm ${subset}.tar.gz
    fi  
done

cd s3prl
if [ ! -d "./data/len_for_bucket" ]; then
    python3 preprocess/generate_len_for_bucket.py -i ${LibriSpeech} --n_jobs 16
fi
python3 run_downstream.py -m train -n ${name}_asr -u customized_upstream -d asr -k ${ckpt} -g ${config} -f \
    -o "config.downstream_expert.datarc.libri_root='${LibriSpeech}',,config.downstream_expert.datarc.bucket_file='./data/len_for_bucket'"
