import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from transformers import BitsAndBytesConfig
import argparse

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

    model.eval()
    model = model.merge_and_unload()

    def gen_prompt(s):
        return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: 請修正文法錯誤：{s} ASSISTANT:"
    res = []
    for d in tqdm(data):
        instruction = gen_prompt(d["instruction"])
        tokenized = tokenizer(instruction, add_special_tokens=False, return_tensors = "pt")
        generate_ids = model.generate(input_ids=tokenized["input_ids"].to("cuda"))
        decoded_pred = tokenizer.batch_decode(generate_ids, skip_special_tokens = False,clean_up_tokenization_spaces=False)[0]
        output = decoded_pred.split("ASSISTANT:")[-1][:-4].replace(" ", "").replace("<s>", "")
        if (args.verbose):
            print(decoded_pred)
            print()
        res.append({"id": d["id"], "output": output})
    with open(args.output_file, 'w+') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()