## Models

- qlora: 1 epochs, trained with all data from FCGEC
- qlora_3_epoch: 1 epoch, trained with all data from FCGEC
- qlora_FCGEC: 3 epochs, trained with 10,000 samples from FCGEC
- qlora_NLPCC: 3 epochs, trained with 10,000 samples from NLPCC2018

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
