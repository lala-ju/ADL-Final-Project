# The project structure

- Datas
We downloaded the datasets with the script `download_data.sh` and saved the raw data in the folder `./data`.
We then preprocessed our datas with the process scripts. The scripts change the dataset format into one instruction sentence and one output answer. Also, there is an option to process the dataset into Taiwan-LLaMa format which added the system prompt before the instruction. `process_FCGEC.sh` is for the FCGEC datasets. `process_NLPCC.sh` is for the NLPCC dataset. The processed dats are saved in the folder `./processed_data`

- BART model
The finetuned process and result on the BART model are all in `./bart` folder. The detailed information is in the README.md inside the directory.

- mT5 model
The finetuned process and result on the mT5 model are all in `./mT5` folder. The detailed information is in the README.md inside the directory.

- Taiwan LLaMa model
The finetuned process and result on the Taiwan-LLaMa model are all in `./tw_llm` folder. The detailed information is in the README.md inside the directory.

- Metrics
We have two different metrics, exact match and $M^2$ score. They are in the `./metric` folder. Detailed information are also in the directory README.md

- Competition Submissions
The `./competition_submissions` folder contains the predictions and scripts that we postprocessed to satisfied the format of the FCGEC competition.