# Train
```
bash train.sh <train file> <error input column name> <corrected input column name>
```

# Model download
Go to the `./model` folder and run the script based on the training dataset to download the respective model.
FCGEC_10000 means the google/mt5-small model finetuned on the 10000 datas based on FCGEC dataset.
FCGEC_all means the google/mt5-small model finetuned on all the datas in the FCGEC dataset.
NLPCC_10000 means the google/mt5-small model finetuned on the 10000 datas based on NLPCC dataset.
NLPCC_all means the google/mt5-small model finetuned on all the datas in the NLPCC dataset.

# Predict
```
bash predict.sh <predict file> <error input column name> <prediction output filename>
```