import torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset
import json
from peft import PeftModel
from transformers import BitsAndBytesConfig
import argparse
from torch.utils.data import DataLoader
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        # required=True,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        required=True,
        help="Path to output file."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--verbose",
        action="store_true"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA
    model = PeftModel.from_pretrained(model, args.peft_path)

    with open(args.test_data_path, "r") as f:
        data = json.load(f)
        ids = [d["id"] for d in data]
    ids_iter = iter(ids)
        
    accelerator = Accelerator()
    accelerator.wait_for_everyone()

    raw_datasets = load_dataset('json', data_files={"test": args.test_data_path})
    column_names = raw_datasets["test"].column_names
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    def gen_prompt(s):
        return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: 請修正文法錯誤：{s} ASSISTANT:"
    
    def preprocess_function(examples):
        inputs = examples["instruction"]
        inputs = [gen_prompt(inp) for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=384, padding=False, truncation=True)
        # model_inputs["_id"] = examples["id"]
        # print(model_inputs)
        return model_inputs
    
    
    with accelerator.main_process_first():
        # Temporarily set max_target_length for validation.
        valid_dataset = raw_datasets["test"].map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )
    
    valid_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, shuffle=False)
    
    def postprocess_text(preds):
        preds = [re.sub(r".*ASSISTANT:", "", pred) for pred in preds]
        return preds


    model, valid_dataloader = accelerator.prepare(
        model, valid_dataloader
    )
    
    model.eval()
    model = model.merge_and_unload()

    gen_kwargs = {"num_beams": args.num_beams}
    
    f = open(args.output_file, "w+")
    for batch in tqdm(valid_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather_for_metrics(generated_tokens)
            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            decoded_preds = postprocess_text(decoded_preds)
            for p in decoded_preds:
                f.write('{"id": "' + next(ids_iter) + '", "output": "' + p.replace("\n", "") + '"}\n')
                f.flush()
            
    
    # for d in tqdm(data):
    #     with torch.no_grad():
    #         instruction = gen_prompt(d["instruction"])
    #         tokenized = tokenizer(instruction, add_special_tokens=False, return_tensors = "pt")
    #         generate_ids = model.generate(input_ids=tokenized["input_ids"].to("cuda"), **gen_kwargs)
    #         decoded_pred = tokenizer.batch_decode(generate_ids, skip_special_tokens = False,clean_up_tokenization_spaces=False)[0]
    #         output = decoded_pred.split("ASSISTANT:")[-1][:-4].replace(" ", "").replace("<s>", "")
    #         if (args.verbose):
    #             print(decoded_pred)
    #             print()
    #         res.append({"id": d["id"], "output": output})
    # with open(args.output_file, 'w+') as f:
    #     json.dump(res, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()