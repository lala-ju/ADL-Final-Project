python3 process_FCGEC.py --input ./data/FCGEC_train.json --output ./processed_data/FCGEC_train.json
python3 process_FCGEC.py --input ./data/FCGEC_train.json --output ./processed_data/FCGEC_train_tw_llm.json --format tw_llm

python3 process_FCGEC.py --input ./data/FCGEC_valid.json --output ./processed_data/FCGEC_valid.json --valid

python3 process_FCGEC.py --input ./data/FCGEC_test.json --output ./processed_data/FCGEC_test.json --format tw_llm --test