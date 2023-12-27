# Train
```
bash train_FCGEC.sh <GPU index you want>
bash train_FCGECall.sh <GPU index you want>
bash train_NLPCC.sh <GPU index you want>
bash train_NLPCCall.sh <GPU index you want>
```

# Model download
Go to the `./model` folder and run `download.sh`
model_FCGEC means the model finetuned on the 10000 datas based on FCGEC dataset.
model_FCGECall means the model finetuned on all the datas in the FCGEC dataset.
model_NLPCC means the model finetuned on the 10000 datas based on NLPCC dataset.
model_NLPCCall means the model finetuned on all the datas in the NLPCC dataset.

# Predict
```
bash inference_FCGEC.sh <GPU index you want> <model path> <output path> <batch size> // for model training on FCGEC
bash inference_NLPCC.sh <GPU index you want> <model path> <output path> <batch size> // for model training on NLPCC
```