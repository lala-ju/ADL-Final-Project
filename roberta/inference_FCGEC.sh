#!/usr/bin/env bash

epochnum=48
mkdir pd_FCGEC

for((i=0;i<${epochnum};i++))
do
    echo ${i}
    cp -f ./models/model_FCGEC01/config.json ./models/model_FCGEC01/tokenizer.json ./models/model_FCGEC01/tokenizer_config.json ./models/model_FCGEC01/epoch_${i}
    CUDA_VISIBLE_DEVICES=${1} python3 inference.py --cuda ${1} --model ./models/model_FCGEC01/epoch_${i} --data ../process_data/FCGEC_valid.json --output ./pd_FCGEC/pd_${i}.json
done