#!/usr/bin/env bash

epochnum=48
mkdir pd_NLPCC

for((i=0;i<${epochnum};i+=2))
do
    echo ${i}
    cp -f ./models/model_NLPCC01/config.json ./models/model_NLPCC01/tokenizer.json ./models/model_NLPCC01/tokenizer_config.json ./models/model_NLPCC01/epoch_${i}
    CUDA_VISIBLE_DEVICES=1 python3 inference.py --batch ${1} --model ./models/model_NLPCC01/epoch_${i} --input ../processed_data/NLPCC2018_valid.json --output ./pd_NLPCC/pd_${i}.json
done