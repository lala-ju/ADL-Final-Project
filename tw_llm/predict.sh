python3 predict.py \
    --base_model_path ../Taiwan-LLM-7B-v2.0-chat \
    --peft_path ./qlora_FCGEC_all \
    --test_data_path ../processed_data/NLPCC2018_valid.json \
    --output_file ./output_valid_NLPCC_all_data_greedy.json \
    --per_device_eval_batch_size 4\
    --verbose