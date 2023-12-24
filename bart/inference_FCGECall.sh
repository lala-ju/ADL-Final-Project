#!/usr/bin/env bash

# epochnum=48
# mkdir pd_FCGECall

# for((i=0;i<${epochnum};i+=2))
# do
#     echo ${i}
#     cp -f ./models/model_FCGECall01/config.json ./models/model_FCGECall01/tokenizer.json ./models/model_FCGECall01/tokenizer_config.json ./models/model_FCGECall01/epoch_${i}
#     CUDA_VISIBLE_DEVICES=${1} python3 inference.py --cuda ${1} --model ./models/model_FCGECall01/epoch_${i} --data ../processed_data/FCGEC_valid.json --output ./pd_FCGECall/pd_${i}.json
# done

CUDA_VISIBLE_DEVICES=${1} python3 inference.py --cuda ${1} --model ${2} --data ../processed_data/FCGEC_valid.json --output ${3}
