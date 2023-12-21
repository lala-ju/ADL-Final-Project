import argparse
import json
from tqdm import tqdm
from opencc import OpenCC

import random

import uuid

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--input", type=str, required=True, help="input file")
    args.add_argument("--output", type=str, required=True, help="output file")
    args.add_argument("--format", type=str, default=None, help="format")
    args.add_argument("--test", action="store_true")
    args.add_argument("--valid", action="store_true")
    args.add_argument("--max_data", type=int, default=10000)
    return args.parse_args()


def main():
    args = parse_args()
    cc = OpenCC('s2tw')
    random.seed(42069)

    with open(args.input, "r") as f:
        data = json.load(f)
    
    if args.test:
        formatted_data = []
        for d in data:
            ans = d["sentence"]
            formatted_data.append({"id": uuid.UUID(bytes = random.randbytes(16), version=4).hex, "instruction": d["sentence"]})
        with open(args.output, "w+") as f:
            json.dump(formatted_data[:args.max_data], f, indent=4, ensure_ascii=False)
    else:   
        if args.valid:
            formatted_data = []
            for d in data:
                if len(d["answers"]) > 0:
                    formatted_data.append({"id": uuid.UUID(bytes = random.randbytes(16), version=4).hex, "instruction": d["sentence"], "output": d["answers"]})
                else:
                    formatted_data.append({"id": uuid.UUID(bytes = random.randbytes(16), version=4).hex, "instruction": d["sentence"], "output": [d["sentence"]]})
            with open(args.output, "w+") as f:
                json.dump(formatted_data[:args.max_data], f, indent=4, ensure_ascii=False)
        else:
            if args.format == "tw_llm":
                def gen_prompt(s):
                    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: 請修正文法錯誤：{s} ASSISTANT:"

                formatted_data = []
                for d in data:
                    if len(d["answers"]) > 0:
                        formatted_data.append({"id": uuid.UUID(bytes = random.randbytes(16), version=4).hex, "instruction": gen_prompt(d["sentence"]), "output": d["answers"][0]})
                    else:
                        formatted_data.append({"id": uuid.UUID(bytes = random.randbytes(16), version=4).hex, "instruction": gen_prompt(d["sentence"]), "output": d["sentence"]})
                with open(args.output, "w+") as f:
                    json.dump(formatted_data[:args.max_data], f, indent=4, ensure_ascii=False)
            else:    
                formatted_data = []
                for d in data:
                    if len(d["answers"]) > 0:
                        formatted_data.append({"id": uuid.UUID(bytes = random.randbytes(16), version=4).hex, "instruction": d["sentence"], "output": d["answers"][0]})
                    else:
                        formatted_data.append({"id": uuid.UUID(bytes = random.randbytes(16), version=4).hex, "instruction": d["sentence"], "output": d["sentence"]})
                with open(args.output, "w+") as f:
                    json.dump(formatted_data[:args.max_data], f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    main()