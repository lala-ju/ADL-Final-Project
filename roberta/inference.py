from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from tqdm.auto import tqdm
import argparse
import json
import torch

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--batch")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # from transformer code
    accelerator = Accelerator(gradient_accumulation_steps=1)
    data_files = {}
    data_files["validation"] = args.input
    extension = args.input.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    column_names = raw_datasets["validation"].column_names
    prefix = ""
    def preprocess_function(examples): # without label
        inputs = [prefix + doc for doc in examples["instruction"]]
        model_inputs = tokenizer(inputs, max_length=int(args.max_src_len), truncation=True)
        return model_inputs
    
    with accelerator.main_process_first():
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
        )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=int(args.batch))
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    num_update_steps = len(eval_dataloader)
    progress_bar = tqdm(range(num_update_steps), disable=not accelerator.is_local_main_process)

    all_result = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                max_length=512,
                do_sample=False,
                num_beams=4,
                attention_mask=batch["attention_mask"]
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = generated_tokens.cpu().numpy()
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for i in range(len(decoded_preds)):
                result = {"id": raw_datasets["validation"][int(args.batch) * step + i]['id'], "output": decoded_preds[i]}
                all_result.append(result)
            progress_bar.update(1)

    with open(args.output, mode='w') as out_file:
        json.dump(all_result, out_file)

if __name__ == "__main__":
    main()