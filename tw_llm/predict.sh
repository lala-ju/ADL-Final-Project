python3 predict.py \
    --base_model_path ../Taiwan-LLM-7B-v2.0-chat \
    --peft_path ./qlora_3_epoch \
    --test_data_path ../processed_data/FCGEC_valid.json \
    --output_file ./output_valid_FCGEC_all_data_greedy.json \
    --per_device_eval_batch_size 4\
    --verbose