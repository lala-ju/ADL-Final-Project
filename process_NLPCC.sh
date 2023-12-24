python3 process_NLPCC.py --input ./data/NLPCC2018_train.json --output ./processed_data/NLPCC2018_train.json
python3 process_NLPCC.py --input ./data/NLPCC2018_train.json --output ./processed_data/NLPCC2018_train_tw_llm.json --format tw_llm

python3 process_NLPCC.py --input ./data/NLPCC2018_train.json --output ./processed_data/NLPCC2018_train_all.json --max_data 100000
python3 process_NLPCC.py --input ./data/NLPCC2018_train.json --output ./processed_data/NLPCC2018_train_all_tw_llm.json --format tw_llm --max_data 100000

python3 process_NLPCC.py --input ./data/NLPCC2018_valid.json --output ./processed_data/NLPCC2018_valid.json --valid