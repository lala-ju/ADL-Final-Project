from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse
import torch
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--cuda")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.data, "r", encoding="utf8") as file:
        all_data = json.load(file)
    
    device = torch.device(f'cuda:{args.cuda}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    progress = tqdm(list(range(len(all_data))))

    result = []
    m2_result = []
    for data in all_data:
        temp = {"id": data["id"]}
        m2_temp = {"id": data["id"], "instruction": data["instruction"]}
        inputs = tokenizer(data["instruction"], return_tensors="pt").to(device).input_ids
        outputs = model.generate(inputs, max_length=128, num_beams=4, do_sample=False).to(device)
        temp["output"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
        m2_temp["output"] = [temp["output"]]
        result.append(temp)
        m2_result.append(m2_temp)
        progress.update(1)

    with open(args.output, 'w', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    with open(f"m2_{args.output}", 'w', encoding='utf8') as f:
        json.dump(m2_result, f, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    main()
