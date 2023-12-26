## Models
- qlora_FCGEC: 3 epochs, trained with 10,000 samples from FCGEC
- qlora_FCGEC_all: 3 epochs, trained with all data from FCGEC
- qlora_NLPCC: 3 epochs, trained with 10,000 samples from NLPCC2018
- qlora_NLPCC_all: 3 epochs, trained with all data from NLPCC2018

## Parameters
- lora_r: 16
- lora_alpha: 32
- lora_dropout: 0.1
- target modules: all linear

- gradient_accumulation_steps: 4
- micro_batch_size: 2
- num_epochs: 3
- optimizer: paged_adamw_32bit
- lr_scheduler: cosine
- learning_rate: 0.0002

- load in 4 bit with fp16


## How to train
- Change `base_model: ../Taiwan-LLM-7B-v2.0-chat` in `qlora.yml` to the directory containing Taiwan LLM
- Change `path: ../processed_data/NLPCC2018_train_all_tw_llm.json` in `qlora.yml` to the desired training data set 
- Change `output_dir: ./qlora_NLPCC_all` in `qlora.yml` to the desired output directory

## How to predict
- Set the fields in `predict.sh` accordingly:
    - `base_model_path`: Taiwan LLM path
    - `peft_path`: qlora module path
    - `test_data_path`: test/valid data path
    - `output_file`: prediction output path
    - `per_device_eval_batch_size`: evaluation batch size