import argparse
import json
from tqdm import tqdm
from opencc import OpenCC

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--input", type=str, required=True, help="input file")
    args.add_argument("--output", type=str, required=True, help="output file")
    args.add_argument("--format", type=str, default=None, help="format")
    return args.parse_args()

def apply_operation(sentence, operation):
    corrected_sentence = []
    for opr in json.loads(operation):
        s = sentence
        for op_type, op_ins in opr.items():
            if op_type == "Switch":
                op_ins = list(map(int, op_ins))
                s = "".join(s[idx] for idx in op_ins)
            elif op_type == "Delete":
                op_ins = set(map(int, op_ins))
                s = "".join(c for idx, c in enumerate(s) if idx not in op_ins)
            elif op_type == "Insert":
                op_ins = op_ins[0]
                s = sentence[:int(op_ins["pos"])] + op_ins["label"] + sentence[int(op_ins["pos"]):]
            elif op_type == "Modify":
                op_ins = op_ins[0]
                d = len(op_ins["label"])
                if len(op_ins["tag"].split("+")) > 1:
                    tag = op_ins["tag"].split("+")[1]
                    if tag[:3] == "INS":
                        d += int(tag[4:])
                    else:
                        d -= int(tag[4:])
                s = sentence[:int(op_ins["pos"])] + op_ins["label"] + sentence[int(op_ins["pos"]) + len(op_ins["label"]):]
            else:
                raise NotImplementedError
        corrected_sentence.append(s)
    return corrected_sentence

def main():
    args = parse_args()
    cc = OpenCC('s2tw')

    with open(args.input, "r") as f:
        data = json.load(f)
    # key: uid
    # value: {"sentence": sentence, "error_flag": error_flag, "error_type": error_type, "operation": operation}
    processed_data = []
    # {"uid": uid, "sentence": sentence, "error_flag": error_flag, "error_type": error_type, "operation": operation, "corrected_sentence": corrected_sentence}
    for uid, prob in tqdm(data.items()):
        corrected_sentence = apply_operation(prob["sentence"], prob["operation"])
        processed_data.append({"uid": uid, "sentence": cc.convert(prob["sentence"]), "error_flag": prob["error_flag"], "corrected_sentence": list(map(cc.convert, corrected_sentence))})


    if args.format == "tw_llm":
        def gen_prompt(s):
            return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: 請修正文法錯誤：{s} ASSISTANT:"

        formatted_data = []
        for d in processed_data:
            if int(d["error_flag"]):
                ans = d["corrected_sentence"][0]
            else:
                ans = d["sentence"]
            formatted_data.append({"id": d["uid"], "instruction": gen_prompt(d["sentence"]), "output": ans})
        with open(args.output, "w+") as f:
            json.dump(formatted_data, f, indent=4, ensure_ascii=False)
    else:    
        with open(args.output, "w+") as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    main()